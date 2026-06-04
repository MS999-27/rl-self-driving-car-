import os
import numpy as np
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.floatlayout import FloatLayout
from kivy.graphics import Color, Ellipse, Line, Rectangle, PushMatrix, PopMatrix, Rotate
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock
from kivy.core.window import Window
from ai import SAC

# --- MACBOOK AIR RETINA & WINDOW FIX ---
from kivy.metrics import Metrics
Metrics.dpi = 96 
Config.set('graphics', 'width', '1000')
Config.set('graphics', 'height', '700')
Config.set('graphics', 'resizable', '0')

brain = SAC(5)
last_reward, penalty_count = 0, 0
scores, entropy_history = [], []
start_time = time.time()
first_update = True

def init(w, h):
    global sand, goal_x, goal_y, first_update
    sand = np.zeros((int(w) + 1, int(h) + 1))
    goal_x, goal_y = 100, h - 100
    first_update = False

class Car(Widget):
    angle = NumericProperty(0)
    velocity = ReferenceListProperty(NumericProperty(0), NumericProperty(0))

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size = (35, 18)
        with self.canvas:
            PushMatrix()
            self.rot = Rotate(angle=self.angle, origin=self.center)
            Color(0, 0.9, 1, 1) # Cyan Car
            self.rect = Rectangle(pos=self.pos, size=self.size)
            PopMatrix()
            Color(1, 0, 0, 1); self.s1 = Ellipse(size=(12, 12))
            Color(0, 1, 0, 1); self.s2 = Ellipse(size=(12, 12))
            Color(0, 0, 1, 1); self.s3 = Ellipse(size=(12, 12))

    def move(self, rotation):
        self.pos = Vector(*self.velocity) + self.pos
        self.angle += rotation
        self.rot.angle, self.rot.origin = self.angle, self.center
        self.rect.pos = self.pos
        
        v_fwd = Vector(55, 0).rotate(self.angle)
        v1, v2, v3 = v_fwd + self.center, v_fwd.rotate(35) + self.center, v_fwd.rotate(-35) + self.center
        self.s1.pos, self.s2.pos, self.s3.pos = (v1-Vector(6,6)), (v2-Vector(6,6)), (v3-Vector(6,6))

        signals = []
        for v in [v1, v2, v3]:
            vx, vy = int(v[0]), int(v[1])
            if 10 < vx < Window.width-10 and 10 < vy < Window.height-10:
                signals.append(float(np.sum(sand[vx-10:vx+10, vy-10:vy+10])) / 400.0)
            else: signals.append(1.0)
        return signals

class Painter(Widget):
    def on_touch_down(self, touch):
        with self.canvas:
            Color(1, 1, 0, 0.5)
            touch.ud['line'] = Line(points=(touch.x, touch.y), width=50)
    def on_touch_move(self, touch):
        if touch.button == 'left':
            touch.ud['line'].points += [touch.x, touch.y]
            x, y = int(touch.x), int(touch.y)
            if 0 <= x < Window.width and 0 <= y < Window.height:
                sand[max(0, x-25):min(int(Window.width), x+25), 
                     max(0, y-25):min(int(Window.height), y+25)] = 1

class Simulation(Widget):
    car_node = ObjectProperty(None)
    def update(self, dt):
        global last_reward, penalty_count, goal_x, goal_y
        if first_update: init(Window.width, Window.height)
        
        sig = self.car_node.move(0) 
        xx, yy = goal_x - self.car_node.x, goal_y - self.car_node.y
        orientation = Vector(*self.car_node.velocity).angle((xx, yy)) / 180.
        rot, ent = brain.update(last_reward, [*sig, orientation, -orientation])
        self.car_node.move(rot)

        # BOUNDARY & COLLISION RESET (STOPS VIBRATING)
        cx, cy = int(self.car_node.center_x), int(self.car_node.center_y)
        if not (20 < cx < Window.width-20 and 20 < cy < Window.height-20) or sand[cx, cy] > 0:
            self.car_node.center = (Window.width/2, Window.height/2)
            self.car_node.velocity = Vector(5, 0).rotate(np.random.randint(0, 360))
            last_reward, penalty_count = -100, penalty_count + 1
        else:
            self.car_node.velocity = Vector(5.5, 0).rotate(self.car_node.angle)
            last_reward = 1.5 if np.sqrt(xx**2 + yy**2) < 400 else -0.5

        if np.sqrt(xx**2 + yy**2) < 70:
            goal_x, goal_y = Window.width - goal_x, Window.height - goal_y
        
        scores.append(last_reward)
        entropy_history.append(ent)

class FinalApp(App):
    def build(self):
        root = FloatLayout()
        with root.canvas.before:
            Color(0.1, 0.1, 0.15, 1); Rectangle(size=(2000, 2000))
        root.add_widget(Painter())
        self.sim = Simulation(); self.car = Car()
        self.sim.add_widget(self.car); self.sim.car_node = self.car
        root.add_widget(self.sim)
        
        self.ui = Label(text="Initializing...", font_size='20sp', bold=True, 
                        pos_hint={'center_x': .18, 'center_y': .9}, color=(0, 1, 1, 1))
        root.add_widget(self.ui)
        
        # SAVE REPORT BUTTON
        btn = Button(text="SAVE REPORT", size_hint=(None, None), size=(180, 50), 
                     pos=(30, 30), background_color=(0, 0.7, 1, 1))
        btn.bind(on_release=self.generate_graph); root.add_widget(btn)
        
        Clock.schedule_interval(self.refresh, 0.1); Clock.schedule_interval(self.sim.update, 1.0/60.0)
        return root

    def refresh(self, dt):
        m, s = divmod(int(time.time() - start_time), 60)
        self.ui.text = f"REWARD: {round(last_reward, 1)}\nPENALTIES: {penalty_count}\nTIME: {m:02d}:{s:02d}"

    def generate_graph(self, instance):
        try:
            plt.style.use('dark_background')
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            data_size = min(len(scores), 2000)
            ax1.plot(scores[-data_size:], color='cyan'); ax1.set_title("Training Rewards")
            ax2.plot(entropy_history[-data_size:], color='magenta'); ax2.set_title("Model Exploration")
            plt.tight_layout()
            
            # --- SAVE TO DESKTOP ---
            desktop = os.path.join(os.path.expanduser("~"), "Desktop")
            save_path = os.path.join(desktop, f"AI_Report_{int(time.time())}.png")
            plt.savefig(save_path)
            plt.close()
            print(f"SUCCESS: Report saved to {save_path}")
            self.ui.text = "REPORT SAVED\nTO DESKTOP!"
        except Exception as e:
            print(f"Error: {e}")

if __name__ == '__main__':
    FinalApp().run()