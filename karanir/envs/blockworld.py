# -*- coding: utf-8 -*-
# @Author: Meleko
# @Date:   2023-10-25 05:28:07
# @Last Modified by:   melkor
# @Last Modified time: 2023-11-11 08:11:27

import arcade
import pymunk
import timeit
import math
import numpy as np

SCREEN_WIDTH =  480
SCREEN_HEIGHT = 360
SCREEN_TITLE = "Shadows of Angrathar"

class PhysicsSprite(arcade.Sprite):
    def __init__(self, pymunk_shape, filename):
        super().__init__(filename, center_x=pymunk_shape.body.position.x, center_y=pymunk_shape.body.position.y)
        self.pymunk_shape = pymunk_shape
        self.file_name = filename


class CircleSprite(PhysicsSprite):
    def __init__(self, pymunk_shape, filename):
        super().__init__(pymunk_shape, filename)
        self.width = pymunk_shape.radius * 2
        self.height = pymunk_shape.radius * 2


class BoxSprite(PhysicsSprite):
    def __init__(self, pymunk_shape, filename, width, height):
        super().__init__(pymunk_shape, filename)
        self.width = width
        self.height = height


def make_body(env, name, x, y, friction = 43.9, mass = 30.0, box_shape = None):
    
    if box_shape is None:
        box_shape = (128,128)
    moment = pymunk.moment_for_box(mass, box_shape)
    body = pymunk.Body(mass, moment)
    body.position = pymunk.Vec2d(x, y)
    shape = pymunk.Poly.create_box(body, box_shape)
    shape.elasticity = 0.001
    shape.friction = 43.9
    env.space.add(body, shape)

    sprite = BoxSprite(shape, "/Users/melkor/Documents/datasets/PatchWork/{}".format(name), width=box_shape[0], height=box_shape[1])
    env.sprite_list.append(sprite)

def make_item(env, name, x, y, friction = 43.9, mass = 30.0, box_shape = None):
    if box_shape is None:
        box_shape = (64,64)
    moment = pymunk.moment_for_box(mass, box_shape)
    body = pymunk.Body(mass, moment)
    body.position = pymunk.Vec2d(x, y)
    shape = pymunk.Poly.create_box(body, box_shape)
    shape.elasticity = 0.001
    shape.friction = 43.9
    env.space.add(body, shape)

    sprite = BoxSprite(shape, "/Users/melkor/Documents/datasets/Acherus/lotrtextures/items/{}.png".format(name), width=box_shape[0], height=box_shape[1])
    env.sprite_list.append(sprite)

def make_block(env, name, x, y, friction = 43.9, mass = 30.0, box_shape = None):
    if box_shape is None:
        box_shape = (64,64)
    moment = pymunk.moment_for_box(mass, box_shape)
    body = pymunk.Body(mass, moment)
    body.position = pymunk.Vec2d(x, y)
    shape = pymunk.Poly.create_box(body, box_shape)
    shape.elasticity = 0.001
    shape.friction = 43.9
    env.space.add(body, shape)

    sprite = BoxSprite(shape, "/Users/melkor/Documents/datasets/Acherus/lotrtextures/blocks/{}.png".format(name), width=box_shape[0], height=box_shape[1])
    env.sprite_list.append(sprite)

class Game(arcade.Window):
    """ Main application class. """

    def __init__(self, width, height, title):
        super().__init__(width, height, title)

        arcade.set_background_color((61,77,86))
        self.paused = False
        self.hold = 0

        # grabber 
        self.graber_x = 200.0
        self.graber_y = 200.0
        self.graber_angl = 0.0

        # -- Pymunk
        self.space = pymunk.Space()
        self.space.iterations = 35
        self.space.gravity = (0.0, -1000.0)

        # Lists of sprites or lines
        self.sprite_list: arcade.SpriteList[PhysicsSprite] = arcade.SpriteList()
        self.static_lines = []

        # Used for dragging shapes around with the mouse
        self.shape_being_dragged = None
        self.last_mouse_position = 0, 0

        self.draw_time = 0
        self.processing_time = 0

        # Create the floor
        floor_height = 80
        size = 64

        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        shape = pymunk.Segment(body, [0, floor_height], [SCREEN_WIDTH, floor_height], 0.0)
        shape.friction = 20
        self.space.add(shape, body)
        self.static_lines.append(shape)

        # Create the stacks of boxes

        self.last_checkpoint_time = timeit.default_timer()
        self.time_diff = 0.2
        self.history_states = []
        self.threshold = 100

    def on_draw(self):
        """
        Render the screen.
        """

        # This command has to happen before we start drawing
        self.clear()

        # Start timing how long this takes
        draw_start_time = timeit.default_timer()

        # Draw all the sprites
        self.sprite_list.draw()

        # Draw the lines that aren't sprites
        for line in self.static_lines:
            body = line.body

            pv1 = body.position + line.a.rotated(body.angle)
            pv2 = body.position + line.b.rotated(body.angle)
            arcade.draw_line(pv1.x, pv1.y, pv2.x, pv2.y, (39,45,52), 2)
        #arcade.draw_rectangle_filled(self.graber_x,self.graber_y, 64, 64, [0,200,0])

        # Display timings
        display_time = False
        if display_time:
            output = f"Processing time: {self.processing_time:.3f}"
            arcade.draw_text(output, 20, SCREEN_HEIGHT - 20, arcade.color.WHITE, 12)

            arcade.draw_text("+", self.graber_x,self.graber_y, arcade.color.WHITE, 12)

            output = f"Drawing time: {self.draw_time:.3f}"
            arcade.draw_text(output, 20, SCREEN_HEIGHT - 40, arcade.color.WHITE, 12)

        self.draw_time = timeit.default_timer() - draw_start_time
        

    def on_mouse_press(self, x, y, button, modifiers):
        if button == arcade.MOUSE_BUTTON_LEFT:
            self.last_mouse_position = x, y
            # See if we clicked on anything
            shape_list = self.space.point_query((x, y), 1, pymunk.ShapeFilter())

            # If we did, remember what we clicked on
            if len(shape_list) > 0:
                self.shape_being_dragged = shape_list[0]

        elif button == arcade.MOUSE_BUTTON_RIGHT:
            # With right mouse button, shoot a heavy coin fast.
            mass = 999
            radius = 20
            inertia = pymunk.moment_for_circle(mass, 0, radius, (0, 0))
            body = pymunk.Body(mass, inertia)
            body.position = x, y
            body.velocity = 100, 10
            shape = pymunk.Circle(body, radius, pymunk.Vec2d(0, 0))
            shape.friction = 9999999.1
            self.space.add(body, shape)

            sprite = CircleSprite(shape, "assets/images/ice.png")
            self.sprite_list.append(sprite)

    def on_mouse_release(self, x, y, button, modifiers):
        if button == arcade.MOUSE_BUTTON_LEFT:
            # Release the item we are holding (if any)
            self.shape_being_dragged = None

    def on_mouse_motion(self, x, y, dx, dy):
        if self.shape_being_dragged is not None:
            # If we are holding an object, move it with the mouse
            self.last_mouse_position = x, y
            self.shape_being_dragged.shape.body.position = self.last_mouse_position
            self.shape_being_dragged.shape.body.velocity = dx * 20, dy * 20

    def save_states(self):
        threshold = self.threshold
        nullary_predicates = []
        unary_predicates = []
        binary_predicates = []
        edges = []

        for sprite in self.sprite_list:
            x, y = sprite.position
            angle = sprite.angle
            name = sprite.file_name.split("/")[-1].split(".")[0]
            category = self.category_map[name]
            unary_predicates.append(np.array([x, y, angle, category]))

        # number of objects in the scene
        num_obj = len(self.sprite_list)

        dist_matrix = np.zeros([num_obj, num_obj])
        adjs_matrix = np.zeros([num_obj, num_obj])
        for i,sprite_x in enumerate(self.sprite_list):
            x1, y1 = sprite_x.position
            for j,sprite_y in enumerate(self.sprite_list):
                x2, y2 = sprite_y.position
                dist = np.linalg.norm(np.array([x1,y1]) - np.array([x2,y2]))
                dist_matrix[i][j] = dist
                if dist < threshold: # add an edge in the binary predicates
                    edges.append([i,j])
                    adjs_matrix[i][j] = 1.0
        binary_predicates.append(dist_matrix)
        binary_predicates.append(adjs_matrix)

        # cast predicates to numpy file
        nullary_predicates = np.array(nullary_predicates)
        unary_predicates = np.array(unary_predicates)
        binary_predicates = np.array(binary_predicates)
        self.history_states.append(
            [nullary_predicates, unary_predicates, binary_predicates]
        )


    def on_key_press(self, symbol, modifiers):
        """Handle user keyboard input
        Q: Quit the game
        P: Pause/Unpause the game
        I/J/K/L: Move Up, Left, Down, Right
        Arrows: Move Up, Left, Down, Right

        Arguments:
            symbol {int} -- Which key was pressed
            modifiers {int} -- Which modifiers were pressed
        """
        if symbol == arcade.key.Q:
            # Quit immediately
            states = self.history_states
            np.save("states",states)
            arcade.close_window()
        step = 15
        if symbol == arcade.key.P:
            self.paused = not self.paused

        if symbol == arcade.key.I or symbol == arcade.key.UP:
            self.graber_y += step

        if symbol == arcade.key.K or symbol == arcade.key.DOWN:
            self.graber_y -= step

        if symbol == arcade.key.J or symbol == arcade.key.LEFT:
            self.graber_x -= step

        if symbol == arcade.key.L or symbol == arcade.key.RIGHT:
            self.graber_x += step

    def on_update(self, delta_time):

        # Check for balls that fall off the screen
        for sprite in self.sprite_list:
            if sprite.pymunk_shape.body.position.y < 0:
                # Remove balls from physics space
                self.space.remove(sprite.pymunk_shape, sprite.pymunk_shape.body)
                # Remove balls from physics list
                sprite.remove_from_sprite_lists()

        # Update physics
        # Use a constant time step, don't use delta_time
        # See "Game loop / moving time forward"
        # https://www.pymunk.org/en/latest/overview.html#game-loop-moving-time-forward
        self.space.step(1 / 60.0)

        # If we are dragging an object, make sure it stays with the mouse. Otherwise
        # gravity will drag it down.
        if self.shape_being_dragged is not None:
            self.shape_being_dragged.shape.body.position = self.last_mouse_position
            self.shape_being_dragged.shape.body.velocity = 0, 0

        # Move sprites to where physics objects are
        for sprite in self.sprite_list:
            sprite.center_x = sprite.pymunk_shape.body.position.x
            sprite.center_y = sprite.pymunk_shape.body.position.y
            sprite.angle = math.degrees(sprite.pymunk_shape.body.angle)
        
        action_sprite = self.sprite_list[0]
        action_sprite.center_x = action_sprite.pymunk_shape.body.position.x
        action_sprite.center_y = action_sprite.pymunk_shape.body.position.y
        action_sprite.angle = math.degrees(action_sprite.pymunk_shape.body.angle)


        # Save the time it took to do this.
        diff = self.processing_time = timeit.default_timer() - self.last_checkpoint_time
        if diff > self.time_diff:
            self.save_states()
            self.last_checkpoint_time = timeit.default_timer()
        
    def set_checkpoint_timediff(self, time_diff):
        """ set the time difference to save the states
        """
        self.time_diff = time_diff

class BlockWorldEnv(Game):
    def __init__(self,width, height, title):
        super().__init__(width, height, title)
        arcade.set_background_color((0.0,10.0,20.0))
        make_block(self, "iccblock", 110,80 + 32);
        make_block(self, "icccraft", 130,80 + 64 + 32);
        make_block(self, "iccblock", 280,80 + 32);
        make_block(self, "icctop", 180,80 + 32);

        dx = (np.random.random() - 0.5) * 90
        make_block(self, "icctop", 150 + dx,120 + 32 + 64*2);

        # [Setup the Save States config]
        self.category_map = {
            "iccblock":1,
            "icccraft":2,
            "icctop":3,
        }
        self.set_checkpoint_timediff(0.2)

def main():
    game = BlockWorldEnv(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)

    game.save_states()
    arcade.run()


if __name__ == "__main__":
    main()