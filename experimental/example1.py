import pygame
import Box2D
import math
from enum import Enum, auto, IntFlag
from collections import deque

# ==============================================================================
# SECTION 1: CONFIGURATION AND CONSTANTS
# ==============================================================================

class Config:
    """Central configuration for the simulation."""
    # Display
    SCREEN_WIDTH = 1280
    SCREEN_HEIGHT = 720
    FPS = 60
    PPM = 20.0  # Pixels per meter
    
    # Physics
    TIME_STEP = 1.0 / 60.0
    VELOCITY_ITERATIONS = 6
    POSITION_ITERATIONS = 2
    
    # Car physics
    MAX_FORWARD_SPEED = 250.0
    MAX_BACKWARD_SPEED = -40.0
    MAX_DRIVE_FORCE = 300.0
    MAX_LATERAL_IMPULSE = 3.0
    REAR_LATERAL_IMPULSE = 4.0
    LOCK_ANGLE_DEG = 40
    TURN_SPEED_DEG_PER_SEC = 320
    
    # Colors (R, G, B, A)
    COLOR_BACKGROUND = (30, 30, 30)
    COLOR_CAR = (200, 100, 50)
    COLOR_TIRE = (50, 50, 50)
    COLOR_MUD = (89, 69, 53, 150)
    COLOR_ICE = (174, 214, 220, 150)
    COLOR_TEXT = (255, 255, 255)
    COLOR_SKID = (20, 20, 20, 100)
    COLOR_STEERING_INDICATOR = (255, 255, 0)
    COLOR_SPEED_BAR = (0, 255, 0)
    COLOR_SPEED_BAR_BG = (50, 50, 50)

# ==============================================================================
# SECTION 2: CORE PHYSICS AND DATA CLASSES
# ==============================================================================

class ControlAction(IntFlag):
    """Enumeration for player control actions using bit flags."""
    LEFT  = 0x1
    RIGHT = 0x2
    UP    = 0x4
    DOWN  = 0x8

class FixtureUserDataType(Enum):
    """Defines the types of fixture user data."""
    CAR_TIRE = auto()
    GROUND_AREA = auto()

class FixtureUserData:
    """A base class for all fixture user data."""
    def __init__(self, type, name=""):
        self.type = type
        self.name = name

class CarTireFUD(FixtureUserData):
    """User data for a car tire fixture."""
    def __init__(self):
        super().__init__(FixtureUserDataType.CAR_TIRE, name="tire")

class GroundAreaFUD(FixtureUserData):
    """User data for a ground area sensor fixture."""
    def __init__(self, friction_modifier, out_of_course, name="ground"):
        super().__init__(FixtureUserDataType.GROUND_AREA, name=name)
        self.friction_modifier = friction_modifier
        self.out_of_course = out_of_course

# ==============================================================================
# SECTION 3: ENHANCED TIRE PHYSICS
# ==============================================================================

class TDTire:
    """
    Represents a single tire, handling all driving, turning, and friction physics.
    Enhanced with skid detection and performance optimizations.
    """
    def __init__(self, world, is_front=True):
        self.world = world
        self.is_front = is_front
        self.body = world.CreateDynamicBody()
        fixture = self.body.CreatePolygonFixture(box=(0.5, 1.25), density=1)
        fixture.userData = CarTireFUD()
        self.body.userData = self
        
        self.max_forward_speed = Config.MAX_FORWARD_SPEED
        self.max_backward_speed = Config.MAX_BACKWARD_SPEED
        self.max_drive_force = Config.MAX_DRIVE_FORCE
        self.max_lateral_impulse = Config.MAX_LATERAL_IMPULSE if is_front else Config.REAR_LATERAL_IMPULSE

        self.ground_areas = set()
        self.current_traction = 1.0
        self.is_skidding = False
        self.last_skid_position = None

    def destroy(self):
        """Safely destroy the tire body."""
        if self.body and self.world:
            self.world.DestroyBody(self.body)
            self.body = None

    def get_lateral_velocity(self):
        right_normal = self.body.GetWorldVector((1, 0))
        return Box2D.b2Dot(right_normal, self.body.linearVelocity) * right_normal

    def get_forward_velocity(self):
        forward_normal = self.body.GetWorldVector((0, 1))
        return Box2D.b2Dot(forward_normal, self.body.linearVelocity) * forward_normal

    def add_ground_area(self, ground_area_fud):
        self.ground_areas.add(ground_area_fud)
        self._update_traction()

    def remove_ground_area(self, ground_area_fud):
        self.ground_areas.discard(ground_area_fud)
        self._update_traction()

    def _update_traction(self):
        if not self.ground_areas:
            self.current_traction = 1.0
        else:
            # Use minimum traction (most slippery surface)
            self.current_traction = min(ga.friction_modifier for ga in self.ground_areas)

    def update_friction(self):
        max_lateral_impulse = self.max_lateral_impulse * self.current_traction
        lateral_velocity = self.get_lateral_velocity()
        impulse = self.body.mass * -lateral_velocity
        
        # Detect skidding
        self.is_skidding = impulse.length > max_lateral_impulse
        
        if self.is_skidding:
            impulse *= max_lateral_impulse / impulse.length
            self.last_skid_position = self.body.worldCenter.copy()
        
        self.body.ApplyLinearImpulse(impulse, self.body.worldCenter, True)
        
        # Angular friction
        self.body.ApplyAngularImpulse(
            0.1 * self.body.inertia * -self.body.angularVelocity * self.current_traction,
            True
        )

        # Forward drag
        forward_velocity = self.get_forward_velocity()
        forward_normal = forward_velocity.copy()
        current_forward_speed = forward_normal.Normalize()
        drag_force_magnitude = -2 * current_forward_speed
        drag_force = drag_force_magnitude * forward_normal * self.current_traction
        self.body.ApplyForce(drag_force, self.body.worldCenter, True)

    def update_drive(self, control_state):
        desired_speed = 0.0
        if control_state & ControlAction.UP:
            desired_speed = self.max_forward_speed
        elif control_state & ControlAction.DOWN:
            desired_speed = self.max_backward_speed
        else:
            return

        forward_normal = self.body.GetWorldVector((0, 1))
        current_speed = Box2D.b2Dot(self.get_forward_velocity(), forward_normal)
        
        force = 0.0
        if desired_speed > current_speed:
            force = self.max_drive_force * self.current_traction
        elif desired_speed < current_speed:
            force = -self.max_drive_force * self.current_traction
        else:
            return
        self.body.ApplyForce(force * forward_normal, self.body.worldCenter, True)

    def get_speed(self):
        """Returns the forward speed of the tire."""
        velocity = self.get_forward_velocity()
        return velocity.length if Box2D.b2Dot(velocity, self.body.GetWorldVector((0, 1))) > 0 else -velocity.length

# ==============================================================================
# SECTION 4: CONTACT LISTENER
# ==============================================================================

class MyContactListener(Box2D.b2ContactListener):
    def BeginContact(self, contact): 
        self._handle_contact(contact, True)
    
    def EndContact(self, contact): 
        self._handle_contact(contact, False)

    def _handle_contact(self, contact, began):
        fudA = contact.fixtureA.userData
        fudB = contact.fixtureB.userData
        if not (fudA and fudB): 
            return
        
        if isinstance(fudA, CarTireFUD) and isinstance(fudB, GroundAreaFUD):
            self._tire_vs_ground_area(contact.fixtureA, contact.fixtureB, began)
        elif isinstance(fudA, GroundAreaFUD) and isinstance(fudB, CarTireFUD):
            self._tire_vs_ground_area(contact.fixtureB, contact.fixtureA, began)

    def _tire_vs_ground_area(self, tire_fixture, ground_area_fixture, began):
        tire = tire_fixture.body.userData
        ground_area_fud = ground_area_fixture.userData
        if began:
            tire.add_ground_area(ground_area_fud)
        else:
            tire.remove_ground_area(ground_area_fud)

# ==============================================================================
# SECTION 5: ENHANCED CAR CLASS
# ==============================================================================

class TDCar:
    """
    Represents a car with a chassis and four independently simulated tires.
    Enhanced with reset functionality and better state tracking.
    """
    def __init__(self, world, initial_position=(0, 10)):
        self.world = world
        self.initial_position = initial_position
        self.initial_angle = 0
        
        # Create car body
        self.body = world.CreateDynamicBody(position=initial_position)
        vertices = [
            (1.5, 0), (3, 2.5), (2.8, 5.5), (1, 10),
            (-1, 10), (-2.8, 5.5), (-3, 2.5), (-1.5, 0)
        ]
        self.body.CreatePolygonFixture(vertices=vertices, density=0.1)
        self.body.userData = FixtureUserData(None, name="car_chassis")

        # Create tires
        self.tires = []
        self.create_tires()
        
        # Track max speed for UI
        self.max_speed_achieved = 0
        
    def create_tires(self):
        """Creates and attaches four tires to the car."""
        joint_def = Box2D.b2RevoluteJointDef(
            bodyA=self.body, 
            enableLimit=True, 
            lowerAngle=0, 
            upperAngle=0, 
            localAnchorB=(0, 0)
        )

        wheel_anchors = {
            'front_left': (-3, 8.5), 
            'front_right': (3, 8.5),
            'back_left': (-3, 0.85), 
            'back_right': (3, 0.85),
        }

        # Front left tire (steerable)
        tire_fl = TDTire(self.world, is_front=True)
        joint_def.bodyB = tire_fl.body
        joint_def.localAnchorA = wheel_anchors['front_left']
        self.fl_joint = self.world.CreateJoint(joint_def)
        self.tires.append(tire_fl)

        # Front right tire (steerable)
        tire_fr = TDTire(self.world, is_front=True)
        joint_def.bodyB = tire_fr.body
        joint_def.localAnchorA = wheel_anchors['front_right']
        self.fr_joint = self.world.CreateJoint(joint_def)
        self.tires.append(tire_fr)
        
        # Rear tires (non-steerable)
        for side in ['back_left', 'back_right']:
            tire = TDTire(self.world, is_front=False)
            joint_def.bodyB = tire.body
            joint_def.localAnchorA = wheel_anchors[side]
            self.world.CreateJoint(joint_def)
            self.tires.append(tire)
            
    def update(self, control_state):
        """Updates car physics and steering."""
        # Update tire physics
        for tire in self.tires: 
            tire.update_friction()
        for tire in self.tires: 
            tire.update_drive(control_state)

        # Steering
        lock_angle_rad = math.radians(Config.LOCK_ANGLE_DEG)
        turn_speed_per_sec_rad = math.radians(Config.TURN_SPEED_DEG_PER_SEC)
        turn_per_time_step = turn_speed_per_sec_rad / Config.FPS

        desired_angle = 0.0
        if control_state & ControlAction.LEFT: 
            desired_angle = lock_angle_rad
        elif control_state & ControlAction.RIGHT: 
            desired_angle = -lock_angle_rad
            
        angle_now = self.fl_joint.angle
        angle_to_turn = desired_angle - angle_now
        angle_to_turn = max(-turn_per_time_step, min(angle_to_turn, turn_per_time_step))
        new_angle = angle_now + angle_to_turn
        
        self.fl_joint.SetLimits(new_angle, new_angle)
        self.fr_joint.SetLimits(new_angle, new_angle)
        
        # Track max speed
        current_speed = abs(self.get_speed())
        self.max_speed_achieved = max(self.max_speed_achieved, current_speed)
    
    def get_speed(self):
        """Returns the car's forward speed."""
        return self.tires[0].get_speed() if self.tires else 0
    
    def get_steering_angle(self):
        """Returns the current steering angle in degrees."""
        return math.degrees(self.fl_joint.angle) if hasattr(self, 'fl_joint') else 0
    
    def reset(self):
        """Resets the car to its initial position."""
        self.body.position = self.initial_position
        self.body.angle = self.initial_angle
        self.body.linearVelocity = (0, 0)
        self.body.angularVelocity = 0
        
        for tire in self.tires:
            tire.body.linearVelocity = (0, 0)
            tire.body.angularVelocity = 0
            tire.ground_areas.clear()
            tire.current_traction = 1.0
        
        # Reset steering
        if hasattr(self, 'fl_joint'):
            self.fl_joint.SetLimits(0, 0)
            self.fr_joint.SetLimits(0, 0)
    
    def destroy(self):
        """Safely destroys the car and its tires."""
        for tire in self.tires:
            tire.destroy()
        if self.body:
            self.world.DestroyBody(self.body)

# ==============================================================================
# SECTION 6: ENHANCED GAME CLASS WITH VISUALIZATION
# ==============================================================================

class Game:
    def __init__(self):
        # --- Pygame Setup ---
        pygame.init()
        self.screen = pygame.display.set_mode((Config.SCREEN_WIDTH, Config.SCREEN_HEIGHT))
        pygame.display.set_caption("Top-Down Car Physics Simulation - Enhanced")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 24)
        self.small_font = pygame.font.SysFont(None, 18)
        self.running = True

        # --- Physics World Setup ---
        self.world = Box2D.b2World(gravity=(0, 0), doSleep=True)
        self.world.contactListener = MyContactListener()

        # --- Game Objects ---
        self.car = TDCar(self.world)
        self.ground_areas = []
        self._create_ground_areas()
        
        self.control_state = 0
        
        # --- Visual Effects ---
        self.skid_marks = deque(maxlen=500)  # Store recent skid positions
        self.camera_offset = [0, 0]
        self.camera_smoothing = 0.1
        
        # --- Performance ---
        self.fps_history = deque(maxlen=60)

    def _create_ground_areas(self):
        """Creates static sensor bodies for different ground surfaces."""
        # Create a single static body for all ground areas
        ground_body = self.world.CreateStaticBody()
        
        # Mud area (left side)
        mud_vertices = [
            (-35, 5), (-5, 5), (-5, 25), (-35, 25)
        ]
        mud_fixture = ground_body.CreatePolygonFixture(
            vertices=mud_vertices,
            isSensor=True
        )
        mud_fixture.userData = GroundAreaFUD(
            friction_modifier=0.4, 
            out_of_course=False, 
            name="mud"
        )
        self.ground_areas.append(mud_fixture)

        # Ice area (right side)
        ice_vertices = [
            (5, 5), (35, 5), (35, 25), (5, 25)
        ]
        ice_fixture = ground_body.CreatePolygonFixture(
            vertices=ice_vertices,
            isSensor=True
        )
        ice_fixture.userData = GroundAreaFUD(
            friction_modifier=0.1, 
            out_of_course=False, 
            name="ice"
        )
        self.ground_areas.append(ice_fixture)

    def run(self):
        """Main game loop."""
        while self.running:
            dt = self.clock.tick(Config.FPS)
            self.fps_history.append(self.clock.get_fps())
            
            self.handle_input()
            self.update_physics()
            self.update_camera()
            self.update_effects()
            self.draw()
            
        self.cleanup()

    def handle_input(self):
        """Processes user input."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_a or event.key == pygame.K_LEFT:
                    self.control_state |= ControlAction.LEFT
                elif event.key == pygame.K_d or event.key == pygame.K_RIGHT:
                    self.control_state |= ControlAction.RIGHT
                elif event.key == pygame.K_w or event.key == pygame.K_UP:
                    self.control_state |= ControlAction.UP
                elif event.key == pygame.K_s or event.key == pygame.K_DOWN:
                    self.control_state |= ControlAction.DOWN
                elif event.key == pygame.K_r:
                    self.car.reset()
                    self.skid_marks.clear()
                elif event.key == pygame.K_ESCAPE:
                    self.running = False
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_a or event.key == pygame.K_LEFT:
                    self.control_state &= ~ControlAction.LEFT
                elif event.key == pygame.K_d or event.key == pygame.K_RIGHT:
                    self.control_state &= ~ControlAction.RIGHT
                elif event.key == pygame.K_w or event.key == pygame.K_UP:
                    self.control_state &= ~ControlAction.UP
                elif event.key == pygame.K_s or event.key == pygame.K_DOWN:
                    self.control_state &= ~ControlAction.DOWN
    
    def update_physics(self):
        """Updates the car and steps the physics world."""
        self.car.update(self.control_state)
        self.world.Step(Config.TIME_STEP, Config.VELOCITY_ITERATIONS, Config.POSITION_ITERATIONS)
    
    def update_camera(self):
        """Smoothly follows the car with the camera."""
        target_x = self.car.body.position.x * Config.PPM
        target_y = self.car.body.position.y * Config.PPM
        
        self.camera_offset[0] += (target_x - self.camera_offset[0]) * self.camera_smoothing
        self.camera_offset[1] += (target_y - self.camera_offset[1]) * self.camera_smoothing
    
    def update_effects(self):
        """Updates visual effects like skid marks."""
        for tire in self.car.tires:
            if tire.is_skidding and tire.last_skid_position:
                world_pos = tire.last_skid_position
                screen_pos = self.world_to_screen(world_pos)
                self.skid_marks.append(screen_pos)

    def world_to_screen(self, world_pos):
        """Converts world coordinates to screen coordinates."""
        x = world_pos[0] * Config.PPM - self.camera_offset[0] + Config.SCREEN_WIDTH / 2
        y = Config.SCREEN_HEIGHT / 2 - (world_pos[1] * Config.PPM - self.camera_offset[1])
        return (x, y)

    def draw(self):
        """Draws all game objects to the screen."""
        self.screen.fill(Config.COLOR_BACKGROUND)
        
        # Draw skid marks
        self.draw_skid_marks()
        
        # Draw ground areas
        self.draw_ground_areas()
        
        # Draw all physics bodies
        for body in self.world.bodies:
            for fixture in body.fixtures:
                self.draw_fixture(fixture)
        
        # Draw UI elements
        self.draw_ui()
        self.draw_steering_indicator()
        self.draw_speedometer()
        
        pygame.display.flip()

    def draw_fixture(self, fixture):
        """Draws a single Box2D fixture."""
        shape = fixture.shape
        if isinstance(shape, Box2D.b2PolygonShape):
            vertices = [(fixture.body.transform * v) for v in shape.vertices]
            vertices = [self.world_to_screen(v) for v in vertices]
            
            # Determine color
            color = Config.COLOR_TIRE
            if fixture.body == self.car.body:
                color = Config.COLOR_CAR
            
            pygame.draw.polygon(self.screen, color, vertices)
            pygame.draw.polygon(self.screen, (0, 0, 0), vertices, 2)  # Black outline

    def draw_ground_areas(self):
        """Draws the transparent ground areas."""
        surface = pygame.Surface((Config.SCREEN_WIDTH, Config.SCREEN_HEIGHT), pygame.SRCALPHA)
        
        for fixture in self.ground_areas:
            shape = fixture.shape
            vertices = [(fixture.body.transform * v) for v in shape.vertices]
            vertices = [self.world_to_screen(v) for v in vertices]
            
            color = Config.COLOR_MUD if fixture.userData.name == "mud" else Config.COLOR_ICE
            pygame.draw.polygon(surface, color, vertices)
            
            # Draw border
            border_color = (*color[:3], 255)  # Full opacity for border
            pygame.draw.polygon(surface, border_color, vertices, 3)
            
            # Draw label
            center_x = sum(v[0] for v in vertices) / len(vertices)
            center_y = sum(v[1] for v in vertices) / len(vertices)
            label = self.font.render(fixture.userData.name.upper(), True, Config.COLOR_TEXT)
            label_rect = label.get_rect(center=(center_x, center_y))
            surface.blit(label, label_rect)
        
        self.screen.blit(surface, (0, 0))

    def draw_skid_marks(self):
        """Draws skid marks on the road."""
        if len(self.skid_marks) > 1:
            surface = pygame.Surface((Config.SCREEN_WIDTH, Config.SCREEN_HEIGHT), pygame.SRCALPHA)
            for i in range(1, len(self.skid_marks)):
                alpha = int(100 * (i / len(self.skid_marks)))  # Fade older marks
                color = (*Config.COLOR_SKID[:3], alpha)
                pygame.draw.circle(surface, color, 
                                 (int(self.skid_marks[i][0]), int(self.skid_marks[i][1])), 
                                 3)
            self.screen.blit(surface, (0, 0))

    def draw_steering_indicator(self):
        """Draws a visual indicator of the steering angle."""
        # Position at bottom center
        center_x = Config.SCREEN_WIDTH // 2
        center_y = Config.SCREEN_HEIGHT - 100
        radius = 40
        
        # Draw steering wheel background
        pygame.draw.circle(self.screen, Config.COLOR_SPEED_BAR_BG, (center_x, center_y), radius, 2)
        
        # Draw steering angle
        angle = self.car.get_steering_angle()
        angle_rad = math.radians(-angle - 90)  # Adjust for pygame coordinates
        end_x = center_x + radius * math.cos(angle_rad)
        end_y = center_y + radius * math.sin(angle_rad)
        
        pygame.draw.line(self.screen, Config.COLOR_STEERING_INDICATOR, 
                        (center_x, center_y), (end_x, end_y), 3)
        
        # Draw center dot
        pygame.draw.circle(self.screen, Config.COLOR_STEERING_INDICATOR, (center_x, center_y), 5)
        
        # Draw angle text
        angle_text = self.small_font.render(f"Steering: {angle:.1f}°", True, Config.COLOR_TEXT)
        text_rect = angle_text.get_rect(center=(center_x, center_y + radius + 20))
        self.screen.blit(angle_text, text_rect)

    def draw_speedometer(self):
        """Draws a speedometer bar."""
        # Position at bottom left
        bar_x = 50
        bar_y = Config.SCREEN_HEIGHT - 150
        bar_width = 200
        bar_height = 20
        
        # Calculate speed percentage
        current_speed = abs(self.car.get_speed())
        max_speed = Config.MAX_FORWARD_SPEED
        speed_percent = min(current_speed / max_speed, 1.0)
        
        # Draw background bar
        pygame.draw.rect(self.screen, Config.COLOR_SPEED_BAR_BG, 
                        (bar_x, bar_y, bar_width, bar_height))
        
        # Draw speed bar
        fill_width = int(bar_width * speed_percent)
        if current_speed > 0:
            bar_color = Config.COLOR_SPEED_BAR
        else:
            bar_color = (255, 0, 0)  # Red for reverse
        
        pygame.draw.rect(self.screen, bar_color, 
                        (bar_x, bar_y, fill_width, bar_height))
        
        # Draw border
        pygame.draw.rect(self.screen, Config.COLOR_TEXT, 
                        (bar_x, bar_y, bar_width, bar_height), 2)
        
        # Draw speed text
        speed_text = self.font.render(f"Speed: {current_speed:.1f} / {max_speed:.0f}", 
                                     True, Config.COLOR_TEXT)
        self.screen.blit(speed_text, (bar_x, bar_y - 30))
        
        # Draw max speed achieved
        max_text = self.small_font.render(f"Max: {self.car.max_speed_achieved:.1f}", 
                                         True, Config.COLOR_TEXT)
        self.screen.blit(max_text, (bar_x, bar_y + bar_height + 5))

    def draw_ui(self):
        """Draws UI text and information."""
        ui_elements = []
        
        # Controls
        ui_elements.append("Controls: WASD or Arrow Keys to drive, R to reset, ESC to quit")
        
        # Traction info
        traction = self.car.tires[0].current_traction if self.car.tires else 1.0
        surface_name = "Normal"
        if self.car.tires and self.car.tires[0].ground_areas:
            surface_name = next(iter(self.car.tires[0].ground_areas)).name.capitalize()
        ui_elements.append(f"Surface: {surface_name} | Traction: {traction:.2f}")
        
        # Skidding indicator
        any_skidding = any(tire.is_skidding for tire in self.car.tires)
        if any_skidding:
            ui_elements.append("SKIDDING!")
        
        # FPS
        avg_fps = sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0
        ui_elements.append(f"FPS: {avg_fps:.1f}")
        
        # Draw all UI elements
        y_offset = 10
        for text in ui_elements:
            color = Config.COLOR_TEXT
            if "SKIDDING" in text:
                color = (255, 100, 100)  # Red for skidding
            
            text_surface = self.font.render(text, True, color)
            self.screen.blit(text_surface, (10, y_offset))
            y_offset += 25

    def cleanup(self):
        """Cleanup resources before exit."""
        self.car.destroy()
        pygame.quit()


# ==============================================================================
# SECTION 7: MAIN ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    game = Game()
    game.run()