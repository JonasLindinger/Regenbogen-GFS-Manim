from manim import *
import numpy as np
import textwrap

# Render-Beispiel:
# manim -pqh water_droplet_dispersion_manim.py WaterDropletDispersion

config.background_color = BLACK

MONO = "DejaVu Sans Mono"

# --- Prism Constants & Helpers ---
PRISM_AIR_COLOR = "#FFE066"
PRISM_GLASS_COLOR = "#FFB347"
PRISM_COLOR = "#7FDBFF"
C_SPEED = 1.8
GLASS_SPEED = 1.2  # about 0.67c
DT_PRISM = 1 / 30

def cross2(a, b):
    return a[0] * b[1] - a[1] * b[0]
  
def segment_intersection(p, p2, q, q2):
    p = np.array(p[:2], dtype=float)
    p2 = np.array(p2[:2], dtype=float)
    q = np.array(q[:2], dtype=float)
    q2 = np.array(q2[:2], dtype=float)
    r = p2 - p
    s = q2 - q
    denom = cross2(r, s)
    if abs(denom) < 1e-9:
        return None
    qp = q - p
    t = cross2(qp, s) / denom
    u = cross2(qp, r) / denom
    if -1e-9 <= t <= 1 + 1e-9 and -1e-9 <= u <= 1 + 1e-9:
        hit = p + t * r
        return np.array([hit[0], hit[1], 0.0]), float(t)
    return None

def outward_normal(a, b, interior_point):
    edge = np.array(b) - np.array(a)
    candidate = normalize(np.array([edge[1], -edge[0], 0.0]))
    midpoint = (np.array(a) + np.array(b)) / 2
    if np.dot(candidate, np.array(interior_point) - midpoint) > 0:
        candidate *= -1
    return candidate

def refract_prism(direction, normal, n1, n2):
    d = normalize(direction)
    n = normalize(normal)
    cos_i = -np.dot(d, n)
    if cos_i < 0:
        n = -n
        cos_i = -np.dot(d, n)
    eta = n1 / n2
    k = 1 - eta**2 * (1 - cos_i**2)
    if k < 0:
        return d
    t = eta * d + (eta * cos_i - np.sqrt(k)) * n
    return normalize(t)

class PropagatingWavefront(VMobject):
    def __init__(
        self,
        entry_a,
        entry_b,
        exit_a,
        exit_b,
        d_air,
        d_glass,
        d_out,
        center,
        extent=1.4,
        samples=17,
        stroke_width=4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.entry_a = np.array(entry_a)
        self.entry_b = np.array(entry_b)
        self.exit_a = np.array(exit_a)
        self.exit_b = np.array(exit_b)
        self.d_air = normalize(d_air)
        self.d_glass = normalize(d_glass)
        self.d_out = normalize(d_out)
        wave_normal = normalize(np.array([-self.d_air[1], self.d_air[0], 0.0]))
        offsets = np.linspace(-extent / 2, extent / 2, samples)
        self.points_data = [
            {"pos": np.array(center) + off * wave_normal, "state": 0}
            for off in offsets
        ]
        self.set_stroke(PRISM_AIR_COLOR, stroke_width)
        self.rebuild_path()
        self.add_updater(self.advance)

    def rebuild_path(self):
        ordered = sorted(self.points_data, key=lambda item: item["pos"][1])
        self.set_points_as_corners([item["pos"] for item in ordered])

    def velocity(self, state):
        if state == 0:
            return self.d_air * C_SPEED
        if state == 1:
            return self.d_glass * GLASS_SPEED
        return self.d_out * C_SPEED

    def advance_point(self, item, dt):
        remaining = dt
        while remaining > 1e-6:
            state = item["state"]
            vel = self.velocity(state)
            proposed = item["pos"] + vel * remaining
            hit = None
            if state == 0:
                hit = segment_intersection(item["pos"], proposed, self.entry_a, self.entry_b)
                next_state = 1
            elif state == 1:
                hit = segment_intersection(item["pos"], proposed, self.exit_a, self.exit_b)
                next_state = 2
            else:
                next_state = 2

            if hit is None:
                item["pos"] = proposed
                break

            point, t = hit
            item["pos"] = point + 1e-4 * self.velocity(next_state)
            item["state"] = next_state
            remaining *= max(0.0, 1 - t)

    def advance(self, mob, dt):
        dt = min(dt, 1 / 15)
        steps = max(1, int(np.ceil(dt / DT_PRISM)))
        sub_dt = dt / steps
        for _ in range(steps):
            for item in self.points_data:
                self.advance_point(item, sub_dt)
        self.rebuild_path()
        states = [item["state"] for item in self.points_data]
        if max(states) == 0:
            self.set_color(PRISM_AIR_COLOR)
        elif min(states) == 1 and max(states) == 1:
            self.set_color(PRISM_GLASS_COLOR)
        elif min(states) == 2:
            self.set_color(PRISM_AIR_COLOR)
        else:
            self.set_color(average_color(PRISM_AIR_COLOR, PRISM_GLASS_COLOR))

# --- Droplet Constants ---
WATER_A = 1.324
WATER_B = 0.00312
AIR_REFRACTIVE_INDEX = 1.0
RADIUS = 2.2
START_X = -7.2
OUT_LENGTH = 6.4
MAX_INTERNAL_BOUNCES = 2
CENTER = np.array([0.0, 0.0, 0.0])
SCENE_SHIFT = LEFT * 2.9
Y_MAX = RADIUS - 1e-3

SELECTED_COLORS = [
    ("Blau", 450.0),
    ("Grün", 540.0),
    ("Rot", 650.0),
]

def scene_point(point: np.ndarray) -> np.ndarray:
    return point + SCENE_SHIFT

def wavelength_to_rgb(wavelength_nm: float) -> ManimColor:
    w = float(np.clip(wavelength_nm, 380, 780))

    if 380 <= w < 440:
        r = -(w - 440) / (440 - 380)
        g = 0.0
        b = 1.0
    elif 440 <= w < 490:
        r = 0.0
        g = (w - 440) / (490 - 440)
        b = 1.0
    elif 490 <= w < 510:
        r = 0.0
        g = 1.0
        b = -(w - 510) / (510 - 490)
    elif 510 <= w < 580:
        r = (w - 510) / (580 - 510)
        g = 1.0
        b = 0.0
    elif 580 <= w < 645:
        r = 1.0
        g = -(w - 645) / (645 - 580)
        b = 0.0
    else:
        r = 1.0
        g = 0.0
        b = 0.0

    if 380 <= w < 420:
        factor = 0.3 + 0.7 * (w - 380) / (420 - 380)
    elif 420 <= w < 701:
        factor = 1.0
    else:
        factor = 0.3 + 0.7 * (780 - w) / (780 - 701)

    gamma = 0.8
    rgb = np.array([r, g, b]) * factor
    rgb = np.power(np.clip(rgb, 0, 1), gamma)
    return rgb_to_color(tuple(rgb))

def water_refractive_index(wavelength_nm: float) -> float:
    lambda_um = wavelength_nm / 1000.0
    return WATER_A + (WATER_B / (lambda_um * lambda_um))

def normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def reflect(v: np.ndarray, normal: np.ndarray) -> np.ndarray:
    return normalize(v - 2 * np.dot(v, normal) * normal)

def refract(incident: np.ndarray, normal: np.ndarray, n1: float, n2: float):
    incident = normalize(incident)
    normal = normalize(normal)
    eta = n1 / n2
    cos_i = -np.dot(normal, incident)
    sin_t2 = eta * eta * (1.0 - cos_i * cos_i)
    if sin_t2 > 1.0:
        return None
    cos_t = np.sqrt(max(0.0, 1.0 - sin_t2))
    refracted = eta * incident + (eta * cos_i - cos_t) * normal
    return normalize(refracted)

def ray_circle_intersection(start: np.ndarray, direction: np.ndarray, radius: float) -> np.ndarray:
    direction = normalize(direction)
    p = start[:2]
    d = direction[:2]
    b = 2.0 * np.dot(p, d)
    c = np.dot(p, p) - radius * radius
    discriminant = b * b - 4.0 * c
    if discriminant < 0:
        return start

    root = np.sqrt(discriminant)
    t1 = (-b - root) / 2.0
    t2 = (-b + root) / 2.0
    candidates = [t for t in (t1, t2) if t > 1e-5]
    if not candidates:
        return start

    t = min(candidates)
    hit = p + d * t
    return np.array([hit[0], hit[1], 0.0])

def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    a = normalize(v1)
    b = normalize(v2)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return float(np.degrees(np.arccos(dot)))

def trace_ray_bundle(wavelength_nm: float, y_offset: float):
    color = wavelength_to_rgb(wavelength_nm)
    n_water = water_refractive_index(wavelength_nm)

    y = float(np.clip(y_offset, 0.0, Y_MAX))
    start = np.array([START_X, y, 0.0])
    entry = np.array([-np.sqrt(max(RADIUS * RADIUS - y * y, 0.0)), y, 0.0])
    incident = normalize(entry - start)
    entry_normal = normalize(entry - CENTER)

    incoming_segments = [(start, entry, 1.0, color, 5.8)]
    inside_segments = []
    reflection_segments = []
    exit_segments = []
    primary = None

    surface_reflection = reflect(incident, entry_normal)
    reflection_segments.append((entry, entry + surface_reflection * 2.5, 0.18, WHITE, 2.2))

    inside_dir = refract(incident, entry_normal, AIR_REFRACTIVE_INDEX, n_water)
    if inside_dir is None:
        return {
            "color": color,
            "incoming": incoming_segments,
            "inside": inside_segments,
            "reflections": reflection_segments,
            "exits": exit_segments,
            "primary": primary,
        }

    current_start = entry
    current_dir = inside_dir

    for bounce in range(MAX_INTERNAL_BOUNCES):
        hit = ray_circle_intersection(current_start + current_dir * 1e-3, current_dir, RADIUS)
        inside_opacity = max(0.35, 1.0 - bounce * 0.16)
        inside_segments.append((current_start, hit, inside_opacity, color, 4.9))

        normal = -normalize(hit - CENTER)
        exit_dir = refract(current_dir, normal, n_water, AIR_REFRACTIVE_INDEX)
        if exit_dir is not None:
            exit_opacity = max(0.35, 1.0 - bounce * 0.16)
            exit_segments.append((hit, hit + exit_dir * OUT_LENGTH, exit_opacity, color, 5.2))
            if bounce == 1:
                primary = {
                    "point": hit,
                    "dir": exit_dir,
                    "angle_deg": angle_between(exit_dir, LEFT),
                }

        reflected_dir = reflect(current_dir, normal)
        reflection_opacity = max(0.24, 0.8 - bounce * 0.15)
        reflection_segments.append((hit, hit + reflected_dir * 0.58, reflection_opacity, color, 2.7))

        current_start = hit
        current_dir = reflected_dir

    return {
        "color": color,
        "incoming": incoming_segments,
        "inside": inside_segments,
        "reflections": reflection_segments,
        "exits": exit_segments,
        "primary": primary,
    }

def glow_line(start, end, color, base_width=5.0, opacity=1.0, z_index=5):
    start = scene_point(start)
    end = scene_point(end)
    layers = VGroup(
        Line(start, end, stroke_color=color, stroke_width=base_width * 3.0, stroke_opacity=0.06 * opacity),
        Line(start, end, stroke_color=color, stroke_width=base_width * 1.6, stroke_opacity=0.18 * opacity),
        Line(start, end, stroke_color=color, stroke_width=base_width, stroke_opacity=0.98 * opacity),
    )
    layers.set_z_index(z_index)
    return layers

def build_beam_group(wavelength_nm: float, y_offset: float):
    bundle = trace_ray_bundle(wavelength_nm, y_offset)
    group = VGroup()

    for segment_group in (bundle["incoming"], bundle["inside"], bundle["exits"], bundle["reflections"]):
        for start, end, opacity, color, width in segment_group:
            z_index = 6 if width >= 4.0 else 5
            group.add(glow_line(start, end, color, base_width=width, opacity=opacity, z_index=z_index))

    return group

def build_droplet():
    core = Circle(radius=RADIUS, color=BLUE_E).move_to(SCENE_SHIFT)
    core.set_stroke(width=2.0, opacity=0.82)
    core.set_fill(color=BLUE_C, opacity=0.08)

    glow_1 = Circle(radius=RADIUS * 1.01, color=BLUE_B).move_to(SCENE_SHIFT)
    glow_1.set_stroke(width=15, opacity=0.08)

    glow_2 = Circle(radius=RADIUS * 1.03, color=BLUE_A).move_to(SCENE_SHIFT)
    glow_2.set_stroke(width=32, opacity=0.03)

    inner = Circle(radius=RADIUS * 0.94, color=WHITE).move_to(SCENE_SHIFT)
    inner.set_stroke(width=1.2, opacity=0.14)

    highlight = Arc(radius=RADIUS * 0.96, start_angle=PI * 0.12, angle=PI * 0.46, color=WHITE)
    highlight.set_stroke(width=3, opacity=0.18)
    highlight.move_arc_center_to(SCENE_SHIFT + LEFT * 0.08 + UP * 0.10)

    droplet = VGroup(glow_2, glow_1, core, inner, highlight)
    droplet.set_z_index(2)
    return droplet

def build_angle_marker(wavelength_nm: float, y_offset: float):
    data = trace_ray_bundle(wavelength_nm, y_offset)
    primary = data["primary"]

    if primary is None:
        return VGroup()

    color = data["color"]
    point = scene_point(primary["point"])
    outgoing_dir = normalize(primary["dir"])
    anti_dir = normalize(LEFT)

    out_tip = point + outgoing_dir * 1.0
    anti_tip = point + anti_dir * 0.92

    out_line = DashedLine(point, out_tip, dash_length=0.08, color=color)
    out_line.set_stroke(width=1.8, opacity=0.8)

    anti_line = DashedLine(point, anti_tip, dash_length=0.08, color=GREY_B)
    anti_line.set_stroke(width=1.7, opacity=0.6)

    helper_left = Line(point, anti_tip, stroke_opacity=0)
    helper_out = Line(point, out_tip, stroke_opacity=0)

    cross_z = anti_dir[0] * outgoing_dir[1] - anti_dir[1] * outgoing_dir[0]
    if abs(cross_z) > 1e-4:
        arc = Angle(helper_left, helper_out, radius=0.38, color=color)
        arc.set_stroke(width=3.0, opacity=0.95)
        return VGroup(anti_line, out_line, arc).set_z_index(7)

    return VGroup(anti_line, out_line).set_z_index(7)

def compute_angle_curve(wavelength_nm: float, n_steps: int = 80):
    points = []
    for i in range(n_steps + 1):
        y = Y_MAX * i / n_steps
        data = trace_ray_bundle(wavelength_nm, y)
        if data["primary"] is not None:
            points.append((y, data["primary"]["angle_deg"]))
    return points

def build_final_graph(color_name: str, wavelength_nm: float, accent_color: ManimColor):
    samples = compute_angle_curve(wavelength_nm)
    if not samples:
        return VGroup()

    y_vals = [p[1] for p in samples]
    y_min = min(y_vals)
    y_max = max(y_vals)
    y_pad = 0.4

    y_axis_min = np.floor(y_min - y_pad)
    y_axis_max = np.ceil(y_max + y_pad)
    y_tick = round((y_axis_max - y_axis_min) / 4, 1)
    if y_tick <= 0:
        y_tick = 0.5

    axes = Axes(
        x_range=[0, Y_MAX, Y_MAX / 4],
        y_range=[y_axis_min, y_axis_max, y_tick],
        x_length=9.0,
        y_length=5.2,
        axis_config={
            "color": GREY_B,
            "stroke_opacity": 0.7,
            "stroke_width": 2.0,
            "include_numbers": False,
        },
        tips=False,
    )

    x_label = Text("Einfallshöhe y [m]", font=MONO, font_size=24, color=GREY_A)
    x_label.next_to(axes, DOWN, buff=0.45)

    y_label = Text("Ablenkwinkel [°]", font=MONO, font_size=24, color=GREY_A)
    y_label.rotate(PI / 2)
    y_label.next_to(axes, LEFT, buff=0.55)

    title = Text(
        f"{color_name}  —  λ = {int(wavelength_nm)} nm",
        font=MONO,
        font_size=30,
        color=accent_color,
    )
    title.next_to(axes, UP, buff=0.45)

    curve_points = [axes.c2p(x, y) for x, y in samples]
    curve = VMobject(color=accent_color)
    curve.set_points_as_corners(curve_points)
    curve.set_stroke(color=accent_color, width=5.5)

    curve_glow = VMobject(color=accent_color)
    curve_glow.set_points_as_corners(curve_points)
    curve_glow.set_stroke(color=accent_color, width=14, opacity=0.12)

    i_max = int(np.argmax(y_vals))
    i_min = int(np.argmin(y_vals))

    peak_pt = axes.c2p(samples[i_max][0], samples[i_max][1])
    peak_dot = Dot(peak_pt, radius=0.07, color=WHITE).set_z_index(10)
    peak_label = Text(
        f"{samples[i_max][1]:.2f}°",
        font=MONO, font_size=20, color=WHITE,
    ).next_to(peak_dot, UP, buff=0.18)

    min_pt = axes.c2p(samples[i_min][0], samples[i_min][1])
    min_dot = Dot(min_pt, radius=0.07, color=accent_color).set_z_index(10)
    min_label = Text(
        f"{samples[i_min][1]:.2f}°",
        font=MONO, font_size=20, color=accent_color,
    ).next_to(min_dot, DOWN, buff=0.18)

    group = VGroup(axes, curve_glow, curve, title, x_label, y_label,
                   peak_dot, peak_label, min_dot, min_label)
    return group

# --- Intro scenes: material response, phase shift, refraction ---
INTRO_BG = "#0F1117"
INTRO_PRIMARY = "#58C4DD"
INTRO_SECONDARY = "#83C167"
INTRO_ACCENT = "#FFDD57"
INTRO_WARNING = "#FF6B6B"
INTRO_STRUCT = "#94A3B8"
INTRO_WHITEISH = "#E5E7EB"

INTRO_TITLE_SIZE = 34
INTRO_BODY_SIZE = 22
INTRO_LABEL_SIZE = 20
INTRO_SMALL_SIZE = 18


def intro_wrap_text(s, width=40):
    return "\n".join(textwrap.wrap(s, width=width))


def intro_make_title(text):
    t = Text(text, font=MONO, font_size=INTRO_TITLE_SIZE, color=INTRO_WHITEISH, weight=BOLD)
    if t.width > 11.5:
        t.set_width(11.5)
    t.to_edge(UP, buff=0.45)
    return t


def intro_make_note(text, color=INTRO_WHITEISH, width=11.6):
    txt = Text(intro_wrap_text(text, 46), font=MONO, font_size=INTRO_BODY_SIZE, color=color, line_spacing=0.9)
    if txt.width > width:
        txt.set_width(width)
    box = RoundedRectangle(corner_radius=0.15, width=12.2, height=1.35,
                           stroke_color=INTRO_STRUCT, stroke_opacity=0.35,
                           fill_color="#141821", fill_opacity=0.92)
    group = VGroup(box, txt)
    txt.move_to(box.get_center())
    group.to_edge(DOWN, buff=0.38)
    return group


def intro_section_label(text, color=INTRO_PRIMARY):
    t = Text(text, font=MONO, font_size=INTRO_LABEL_SIZE, color=color, weight=BOLD)
    if t.width > 4.2:
        t.set_width(4.2)
    return t


class IntroBaseScene(Scene):
    def setup(self):
        self.camera.background_color = INTRO_BG

    def change_note(self, old_note, text, color=INTRO_WHITEISH):
        if old_note is not None and old_note in self.mobjects:
            self.remove(old_note)
        return None

    def clean_end(self):
        if self.mobjects:
            self.play(FadeOut(Group(*self.mobjects)), run_time=0.5)
            self.wait(0.3)


class Intro1_Leitfrage(IntroBaseScene):
    def construct(self):
        title = intro_make_title("Phasenverschiebung: Was passiert im Material?")
        self.add(title)

        left_box = RoundedRectangle(width=5.8, height=3.7, corner_radius=0.18,
                                    stroke_color=INTRO_STRUCT, stroke_opacity=0.45)
        right_box = left_box.copy()
        left_box.shift(LEFT * 3.45 + UP * 0.15)
        right_box.shift(RIGHT * 3.45 + UP * 0.15)
        left_label = intro_section_label("Vakuum", INTRO_PRIMARY).next_to(left_box, UP, buff=0.16)
        right_label = intro_section_label("Material", INTRO_SECONDARY).next_to(right_box, UP, buff=0.16)

        ax_l = Axes(x_range=[0, 6.2, 1], y_range=[-1.5, 1.5, 1], x_length=4.8, y_length=2.1,
                    axis_config={"color": INTRO_STRUCT, "stroke_opacity": 0.25, "include_tip": False})
        ax_r = ax_l.copy()
        ax_l.move_to(left_box)
        ax_r.move_to(right_box)

        phase = ValueTracker(0)
        wave_l = always_redraw(lambda: ax_l.plot(lambda x: 0.75 * np.sin(2.0 * x - phase.get_value()), color=INTRO_PRIMARY, stroke_width=4))
        wave_r = always_redraw(lambda: ax_r.plot(lambda x: 0.75 * np.sin(2.4 * x - 0.78 * phase.get_value() - 0.8), color=INTRO_ACCENT, stroke_width=4))

        freq_text = Text("gleiche Lichtfrequenz", font=MONO, font_size=16, color=INTRO_WHITEISH)
        freq_text.move_to(DOWN * 1.55)
        caution = Text("Nicht: Stop-and-Go", font=MONO, font_size=INTRO_SMALL_SIZE, color=INTRO_WARNING)
        caution.next_to(freq_text, DOWN, buff=0.16)

        note = None
        note = self.change_note(note, "Frage: Warum ist die Welle im Material phasenverschoben und scheinbar langsamer?")
        self.play(FadeIn(left_box), FadeIn(right_box), FadeIn(left_label), FadeIn(right_label), FadeIn(ax_l), FadeIn(ax_r), run_time=1.0)
        self.wait(0.5)
        self.play(Create(wave_l), Create(wave_r), FadeIn(freq_text), run_time=1.5)
        self.wait(0.8)
        self.play(phase.animate.set_value(4 * PI), run_time=4.0, rate_func=linear)
        self.wait(0.8)
        note = self.change_note(note, "Wichtig: Das Licht wird nicht wie ein Ball immer wieder angehalten. Die Ursache ist die Antwort der Ladungen im Material.")
        self.play(FadeIn(caution), run_time=0.8)
        self.wait(1.2)
        self.clean_end()


class Intro2_ElektronFolgtVerzoegert(IntroBaseScene):
    def construct(self):
        title = intro_make_title("1) Das Lichtfeld treibt ein gebundenes Elektron an")
        self.add(title)
        note = None
        note = self.change_note(note, "Eine Lichtwelle besitzt ein elektrisches Feld. Dieses Feld übt auf geladene Teilchen eine Kraft aus.")

        atom_center = UP * 0.25
        orbit_line = Line(atom_center + LEFT * 2.2, atom_center + RIGHT * 2.2, color=INTRO_STRUCT, stroke_opacity=0.25)
        nucleus = Dot(atom_center, radius=0.18, color=INTRO_WARNING)
        nucleus_label = Text("Atomkern", font=MONO, font_size=INTRO_SMALL_SIZE, color=INTRO_WHITEISH).next_to(nucleus, UP, buff=0.18)
        guide = DashedLine(atom_center + LEFT * 2.0, atom_center + RIGHT * 2.0, color=INTRO_STRUCT, stroke_opacity=0.25)

        t = ValueTracker(0)
        response = lambda: 1.15 * np.sin(t.get_value() - 0.85)

        field_arrow = always_redraw(lambda: Arrow(atom_center + LEFT * 4.8, atom_center + LEFT * 4.8 + RIGHT * (1.2 + 0.9 * np.sin(t.get_value())),
                                                   buff=0, stroke_width=6, max_stroke_width_to_length_ratio=10,
                                                   max_tip_length_to_length_ratio=0.2, color=INTRO_PRIMARY))
        field_label = Text("E-Feld der Lichtwelle", font=MONO, font_size=INTRO_SMALL_SIZE, color=INTRO_PRIMARY)
        field_label.next_to(atom_center + LEFT * 4.0 + UP * 1.3, DOWN, buff=0.1)

        spring = always_redraw(lambda: ParametricFunction(
            lambda u: np.array([
                atom_center[0] + (-0.2 + (response() + 0.2) * u),
                atom_center[1] + 0.18 * np.sin(10 * PI * u),
                0,
            ]),
            t_range=[0, 1], color=INTRO_STRUCT, stroke_width=2.5, stroke_opacity=0.6,
        ))
        electron = always_redraw(lambda: Dot(atom_center + RIGHT * response(), radius=0.14, color=INTRO_SECONDARY))
        electron_label = always_redraw(lambda: Text("Elektron", font=MONO, font_size=INTRO_SMALL_SIZE, color=INTRO_SECONDARY).next_to(electron, DOWN, buff=0.16))
        rest_marker = Dot(atom_center + LEFT * 0.05, radius=0.05, color=INTRO_STRUCT)

        top_hint = Text("Bindung + Trägheit => Nachhinken", font=MONO, font_size=INTRO_SMALL_SIZE, color=INTRO_ACCENT)
        top_hint.move_to(UP * 2.4)

        self.play(FadeIn(orbit_line), FadeIn(guide), FadeIn(nucleus), FadeIn(nucleus_label), FadeIn(rest_marker), run_time=1.0)
        self.wait(0.5)
        self.play(FadeIn(field_arrow), FadeIn(field_label), FadeIn(spring), FadeIn(electron), FadeIn(electron_label), run_time=1.0)
        self.wait(0.8)
        self.play(t.animate.set_value(3 * PI), run_time=4.0, rate_func=linear)
        self.wait(0.8)
        note = self.change_note(note, "Das Elektron folgt nicht exakt sofort. Es ist an das Atom gebunden und hat Trägheit. Deshalb hinkt seine Bewegung dem Feld etwas hinterher.")
        self.play(FadeIn(top_hint), run_time=0.8)
        self.play(t.animate.set_value(6 * PI), run_time=4.0, rate_func=linear)
        self.wait(1.1)
        self.clean_end()


class Intro3_WannStrahltEineLadung(IntroBaseScene):
    def construct(self):
        title = intro_make_title("2) Wann sendet ein Teilchen elektromagnetische Wellen aus?")
        self.add(title)
        note = None
        note = self.change_note(note, "Die kurze Antwort lautet: Wenn eine Ladung beschleunigt wird. Reine Ruhe oder gleichförmige Bewegung strahlen in diesem einfachen Sinn nicht ab.")

        panels = VGroup(*[
            RoundedRectangle(width=3.95, height=4.0, corner_radius=0.16, stroke_color=INTRO_STRUCT, stroke_opacity=0.45)
            for _ in range(3)
        ]).arrange(RIGHT, buff=0.35).shift(UP * 0.45)
        labels = VGroup(
            Text("Ruhe", font=MONO, font_size=INTRO_SMALL_SIZE, color=INTRO_WHITEISH),
            Text("gleichförmig", font=MONO, font_size=INTRO_SMALL_SIZE, color=INTRO_WHITEISH),
            Text("beschleunigt", font=MONO, font_size=INTRO_SMALL_SIZE, color=INTRO_WHITEISH),
        )
        for lbl, panel in zip(labels, panels):
            lbl.next_to(panel, UP, buff=0.12)

        self.play(FadeIn(panels), FadeIn(labels), run_time=1.0)
        self.wait(0.5)

        p1 = panels[0].get_center()
        charge1 = Dot(p1, radius=0.12, color=INTRO_SECONDARY)
        halo1 = Circle(radius=0.55, color=INTRO_PRIMARY, stroke_opacity=0.18).move_to(p1)
        text1 = Text("keine neue\nabgestrahlte\nWelle", font=MONO, font_size=17, color=INTRO_WHITEISH, line_spacing=0.8)
        text1.move_to(panels[0].get_bottom() + UP * 0.72)
        self.play(FadeIn(charge1), FadeIn(halo1), FadeIn(text1), run_time=0.9)
        self.wait(0.8)

        p2 = panels[1].get_center()
        charge2 = Dot(p2 + LEFT * 1.0, radius=0.12, color=INTRO_SECONDARY)
        arrow2 = Arrow(p2 + LEFT * 1.35, p2 + RIGHT * 1.2, buff=0, color=INTRO_PRIMARY, stroke_width=5)
        trail2 = VGroup(*[Circle(radius=r, color=INTRO_PRIMARY, stroke_opacity=0.08).move_to(p2 + RIGHT * (0.4 * r)) for r in [0.35, 0.7, 1.05]])
        text2 = Text("bloße Bewegung\nist noch keine\nStrahlung", font=MONO, font_size=17, color=INTRO_WHITEISH, line_spacing=0.8)
        text2.move_to(panels[1].get_bottom() + UP * 0.72)
        self.play(FadeIn(charge2), GrowArrow(arrow2), FadeIn(trail2), FadeIn(text2), run_time=1.0)
        self.wait(0.8)

        p3 = panels[2].get_center()
        osc = ValueTracker(0)
        charge3 = always_redraw(lambda: Dot(p3 + RIGHT * (0.8 * np.sin(osc.get_value())), radius=0.12, color=INTRO_SECONDARY))
        path3 = Line(p3 + LEFT * 1.1, p3 + RIGHT * 1.1, color=INTRO_STRUCT, stroke_opacity=0.25)
        text3 = Text("Schwingen =\nständige\nBeschleunigung", font=MONO, font_size=17, color=INTRO_WHITEISH, line_spacing=0.8)
        text3.move_to(panels[2].get_bottom() + UP * 0.72)
        ripples = VGroup(*[Circle(radius=0.15, color=INTRO_ACCENT, stroke_opacity=0.55).move_to(p3) for _ in range(3)])
        self.play(FadeIn(path3), FadeIn(charge3), FadeIn(text3), run_time=1.0)
        self.wait(0.6)
        self.play(osc.animate.set_value(4 * PI),
                  *[r.animate.scale(8).set_stroke(opacity=0) for r in ripples],
                  run_time=3.2, rate_func=linear)
        self.wait(0.8)
        note = self.change_note(note, "Warum? Eine schwingende Ladung wird fortwährend beschleunigt. Genau diese zeitlich veränderliche Bewegung erzeugt eine elektromagnetische Strahlung.")
        accel_mark = Text("Beschleunigung => Abstrahlung", font=MONO, font_size=INTRO_SMALL_SIZE, color=INTRO_ACCENT)
        accel_mark.move_to(UP * 2.35)
        self.play(FadeIn(accel_mark), run_time=0.8)
        self.wait(1.2)
        self.clean_end()


class Intro4_VieleAtomePhasenverschiebung(IntroBaseScene):
    def construct(self):
        title = intro_make_title("3) Viele Atome zusammen verschieben die Phase")
        self.add(title)
        note = None
        note = self.change_note(note, "In einem Material schwingen nicht nur ein einziges, sondern sehr viele gebundene Elektronen. Ihre Antwort überlagert sich mit der einfallenden Welle.")

        material_box = RoundedRectangle(width=9.6, height=3.1, corner_radius=0.15,
                                        stroke_color=INTRO_STRUCT, stroke_opacity=0.45,
                                        fill_color="#131A22", fill_opacity=0.35).shift(UP * 0.3)
        material_label = Text("Material", font=MONO, font_size=INTRO_SMALL_SIZE, color=INTRO_SECONDARY).next_to(material_box, UP, buff=0.12)
        atoms = VGroup()
        for x in np.linspace(-4.0, 4.0, 7):
            core = Dot(np.array([x, 0.1, 0]), radius=0.08, color=INTRO_WARNING)
            electron = Dot(np.array([x + 0.22, 0.1, 0]), radius=0.05, color=INTRO_SECONDARY)
            bond = Line(core.get_center(), electron.get_center(), color=INTRO_STRUCT, stroke_opacity=0.35)
            atoms.add(VGroup(core, bond, electron))
        atoms.shift(UP * 0.3)

        ax_top = Axes(x_range=[0, 10, 1], y_range=[-1.4, 1.4, 1], x_length=10.0, y_length=1.4,
                      axis_config={"color": INTRO_STRUCT, "stroke_opacity": 0.18, "include_tip": False})
        ax_mid = ax_top.copy()
        ax_bot = ax_top.copy()
        ax_top.move_to(UP * 2.15)
        ax_mid.move_to(UP * 0.0)
        ax_bot.move_to(DOWN * 1.75)

        incoming_label = Text("einfallende Welle", font=MONO, font_size=INTRO_SMALL_SIZE, color=INTRO_PRIMARY)
        incoming_label.move_to(ax_top.get_left() + RIGHT * 1.55 + UP * 0.55)
        response_label = Text("Materialantwort", font=MONO, font_size=INTRO_SMALL_SIZE, color=INTRO_SECONDARY)
        response_label.move_to(ax_mid.get_left() + RIGHT * 1.6 + UP * 0.55)
        result_label = Text("resultierende Welle", font=MONO, font_size=INTRO_SMALL_SIZE, color=INTRO_ACCENT)
        result_label.move_to(ax_bot.get_left() + RIGHT * 1.75 + UP * 0.55)

        phase = ValueTracker(0)
        incoming = always_redraw(lambda: ax_top.plot(lambda x: 0.55 * np.sin(1.7 * x - phase.get_value()), color=INTRO_PRIMARY, stroke_width=4))
        response = always_redraw(lambda: ax_mid.plot(lambda x: 0.38 * np.sin(1.7 * x - phase.get_value() - 0.95), color=INTRO_SECONDARY, stroke_width=4))
        result = always_redraw(lambda: ax_bot.plot(lambda x: 0.55 * np.sin(1.7 * x - phase.get_value() - 0.42), color=INTRO_ACCENT, stroke_width=4))

        self.play(FadeIn(material_box), FadeIn(material_label), FadeIn(atoms), run_time=1.0)
        self.wait(0.5)
        self.play(FadeIn(ax_top), FadeIn(ax_mid), FadeIn(ax_bot), FadeIn(incoming_label), FadeIn(response_label), FadeIn(result_label), run_time=1.0)
        self.wait(0.4)
        self.play(Create(incoming), Create(response), Create(result), run_time=1.4)
        self.wait(0.8)
        self.play(phase.animate.set_value(4 * PI), run_time=4.0, rate_func=linear)
        self.wait(0.8)

        shift_arrow = DoubleArrow(ax_bot.c2p(4.4, 0.95), ax_bot.c2p(5.0, 0.95), color=INTRO_ACCENT, stroke_width=4)
        shift_text = Text("Phasenverschiebung", font=MONO, font_size=INTRO_SMALL_SIZE, color=INTRO_ACCENT).next_to(shift_arrow, UP, buff=0.12)
        self.play(FadeIn(shift_arrow), FadeIn(shift_text), run_time=0.8)
        self.wait(0.8)
        note = self.change_note(note, "Das Entscheidende ist die Überlagerung: Die vom Material erzeugten Felder addieren sich zur ursprünglichen Welle. Dadurch entsteht eine neue Welle mit verschobener Phase.")
        myth = Text("Nicht Anhalten, sondern Überlagerung", font=MONO, font_size=INTRO_SMALL_SIZE, color=INTRO_WARNING)
        myth.move_to(UP * 2.55 + RIGHT * 2.2)
        self.play(FadeIn(myth), run_time=0.8)
        self.play(phase.animate.set_value(7 * PI), run_time=3.0, rate_func=linear)
        self.wait(1.1)
        self.clean_end()


class Intro5_VonPhaseZuBrechung(IntroBaseScene):
    def construct(self):
        title = intro_make_title("4) An der Grenzfläche wird daraus Brechung")
        self.add(title)
        note = None
        note = self.change_note(note, "Trifft eine Wellenfront schräg auf Wasser, gelangt ihr unterer Teil zuerst in das Medium und wird dort zuerst langsamer.")

        boundary = Line(LEFT * 6.3 + DOWN * 0.45, RIGHT * 6.3 + DOWN * 0.45, color=INTRO_WHITEISH, stroke_width=3)
        air = Text("Luft", font=MONO, font_size=INTRO_SMALL_SIZE, color=INTRO_PRIMARY).move_to(LEFT * 5.4 + UP * 2.5)
        water = Text("Wasser", font=MONO, font_size=INTRO_SMALL_SIZE, color=INTRO_SECONDARY).move_to(LEFT * 5.2 + DOWN * 2.5)
        self.play(Create(boundary), FadeIn(air), FadeIn(water), run_time=1.0)
        self.wait(0.4)

        fronts_top = VGroup(*[
            Line(LEFT * 1.8 + UP * (2.8 - 0.7 * i), RIGHT * 1.8 + UP * (1.7 - 0.7 * i), color=INTRO_PRIMARY, stroke_width=3)
            for i in range(4)
        ]).shift(LEFT * 1.8)
        beam = Arrow(LEFT * 4.4 + UP * 2.3, LEFT * 1.9 + UP * 0.8, buff=0, color=INTRO_PRIMARY, stroke_width=5)
        self.play(LaggedStart(*[Create(f) for f in fronts_top], lag_ratio=0.15), GrowArrow(beam), run_time=1.6)
        self.wait(0.8)

        note = self.change_note(note, "Weil ein Teil der Wellenfront früher im Wasser ist als der andere, dreht sich die ganze Front. Genau diese Drehung ändert die Ausbreitungsrichtung.")
        fronts_bottom = VGroup(*[
            Line(LEFT * 1.0 + DOWN * (0.4 + 0.6 * i), RIGHT * 1.5 + DOWN * (0.9 + 0.6 * i), color=INTRO_ACCENT, stroke_width=3)
            for i in range(4)
        ]).shift(RIGHT * 1.05)
        beam2 = Arrow(LEFT * 1.2 + DOWN * 0.1, RIGHT * 1.0 + DOWN * 1.7, buff=0, color=INTRO_ACCENT, stroke_width=5)
        normal = DashedLine(ORIGIN + UP * 2.2 + RIGHT * 0.2, ORIGIN + DOWN * 2.2 + RIGHT * 0.2, color=INTRO_STRUCT, stroke_opacity=0.4)
        self.play(Create(normal), LaggedStart(*[Create(f) for f in fronts_bottom], lag_ratio=0.15), GrowArrow(beam2), run_time=1.8)
        self.wait(0.8)

        top_dot = Dot(LEFT * 1.8 + UP * 0.75, radius=0.08, color=INTRO_PRIMARY)
        bot_dot = Dot(LEFT * 0.6 + DOWN * 0.8, radius=0.08, color=INTRO_ACCENT)
        early = Text("tritt zuerst ein", font=MONO, font_size=INTRO_SMALL_SIZE, color=INTRO_ACCENT).next_to(bot_dot, RIGHT, buff=0.18)
        self.play(FadeIn(top_dot), FadeIn(bot_dot), FadeIn(early), run_time=0.8)
        self.wait(1.0)
        self.clean_end()


class Intro6_FarbenUndRegenbogen(IntroBaseScene):
    def construct(self):
        title = intro_make_title("5) Verschiedene Farben => verschiedene Brechung")
        self.add(title)
        note = None
        note = self.change_note(note, "Die Materialantwort hängt von der Frequenz ab. Deshalb ist die Phasenverschiebung für Blau und Rot nicht genau gleich groß.")

        source = Dot(LEFT * 5.4 + UP * 0.8, radius=0.12, color=INTRO_WHITEISH)
        source_label = Text("weißes Licht", font=MONO, font_size=INTRO_SMALL_SIZE, color=INTRO_WHITEISH).next_to(source, UP, buff=0.16)
        drop = Circle(radius=1.0, color=INTRO_STRUCT, stroke_width=3, fill_color="#1A2533", fill_opacity=0.35).move_to(ORIGIN + RIGHT * 0.2)
        drop_label = Text("Wassertropfen", font=MONO, font_size=INTRO_SMALL_SIZE, color=INTRO_SECONDARY).next_to(drop, UP, buff=0.16)
        ray_in = Arrow(source.get_center(), drop.get_left() + UP * 0.1, buff=0, color=INTRO_WHITEISH, stroke_width=5)
        ray_red = Arrow(drop.get_right() + DOWN * 0.05, RIGHT * 5.5 + DOWN * 0.45, buff=0, color="#FF6B6B", stroke_width=5)
        ray_blue = Arrow(drop.get_right() + DOWN * 0.05, RIGHT * 5.0 + DOWN * 1.3, buff=0, color="#60A5FA", stroke_width=5)
        red_label = Text("Rot: kleinere Verzögerung", font=MONO, font_size=INTRO_SMALL_SIZE, color="#FF6B6B")
        blue_label = Text("Blau: stärkere Verzögerung", font=MONO, font_size=INTRO_SMALL_SIZE, color="#60A5FA")
        red_label.move_to(RIGHT * 3.1 + UP * 1.8)
        blue_label.move_to(RIGHT * 3.15 + UP * 1.2)

        self.play(FadeIn(source), FadeIn(source_label), FadeIn(drop), FadeIn(drop_label), GrowArrow(ray_in), run_time=1.3)
        self.wait(0.8)
        self.play(GrowArrow(ray_red), GrowArrow(ray_blue), FadeIn(red_label), FadeIn(blue_label), run_time=1.6)
        self.wait(0.8)
        note = self.change_note(note, "Blaues Licht koppelt im Wasser etwas anders an die Elektronen als rotes Licht. Dadurch wird Blau stärker gebrochen. Im Regentropfen trennt das die Farben auf.")

        summary = VGroup(
            Text("Kurzform:", font=MONO, font_size=INTRO_LABEL_SIZE, color=INTRO_ACCENT, weight=BOLD),
            Text("Lichtfeld treibt Elektronen an", font=MONO, font_size=INTRO_SMALL_SIZE, color=INTRO_WHITEISH),
            Text("Elektronen hinken leicht nach", font=MONO, font_size=INTRO_SMALL_SIZE, color=INTRO_WHITEISH),
            Text("beschleunigte Ladungen senden Felder aus", font=MONO, font_size=INTRO_SMALL_SIZE, color=INTRO_WHITEISH),
            Text("Überlagerung verschiebt die Phase", font=MONO, font_size=INTRO_SMALL_SIZE, color=INTRO_WHITEISH),
            Text("an der Grenzfläche entsteht Brechung", font=MONO, font_size=INTRO_SMALL_SIZE, color=INTRO_WHITEISH),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        summary.scale(0.92)
        summary.to_corner(UL, buff=0.65)
        summary.shift(DOWN * 0.55)
        bg = RoundedRectangle(width=5.15, height=2.5, corner_radius=0.16, stroke_color=INTRO_STRUCT, stroke_opacity=0.3,
                              fill_color="#141821", fill_opacity=0.88).move_to(summary.get_center())
        self.play(FadeIn(bg), LaggedStart(*[FadeIn(m, shift=RIGHT * 0.12) for m in summary], lag_ratio=0.12), run_time=1.6)
        self.wait(2.0)
        self.clean_end()

class WaterDropletDispersion(Scene):
    def construct(self):
        # ══════════════════════════════════════════════════════════════════
        # EINLEITUNG: Lichtmodelle
        # ══════════════════════════════════════════════════════════════════
        
        intro_title = Text("Modelle des Lichts", font=MONO, font_size=42, color=WHITE)
        intro_title.to_edge(UP, buff=1.0)
        
        model_wave = Text("1. Wellenmodell (Wellenfronten)", font=MONO, font_size=28, color=PRISM_AIR_COLOR)
        model_ray = Text("2. Teilchenmodell (Lichtstrahlen)", font=MONO, font_size=28, color=YELLOW_B)
        
        models = VGroup(model_wave, model_ray).arrange(DOWN, aligned_edge=LEFT, buff=1.4)
        models.next_to(intro_title, DOWN, buff=1.0)
        
        self.play(Write(intro_title))
        self.wait(0.5)
        self.play(FadeIn(model_wave, shift=RIGHT))
        self.wait(1.5)
        self.play(FadeIn(model_ray, shift=RIGHT))
        self.wait(3.0)
        
        self.play(FadeOut(intro_title), FadeOut(models))
        self.wait(0.5)

        # ══════════════════════════════════════════════════════════════════
        # SZENE 0: Prism Refraction (Wavefronts)
        p1 = np.array([-10.0, -12.0, 0.0])
        p2 = np.array([10.0, 22.64, 0.0])
        p3 = np.array([30.0, -12.0, 0.0])
        prism_center = (p1 + p2 + p3) / 3

        entry_normal = outward_normal(p1, p2, prism_center)
        exit_normal = outward_normal(p2, p3, prism_center)

        d_air = np.array([1.0, 0.0, 0.0])
        d_glass = refract_prism(d_air, entry_normal, 1.0, 1.5)
        d_out = refract_prism(d_glass, exit_normal, 1.5, 1.0)

        prism = Polygon(
            p1, p2, p3,
            fill_color=PRISM_COLOR,
            fill_opacity=0.15,
            stroke_color=PRISM_COLOR,
            stroke_width=2
        )

        fronts = VGroup()
        for i in range(20):
            center = np.array([-7.5 - 0.45 * i, 0.8, 0.0])
            front = PropagatingWavefront(
                p1, p2, p2, p3,
                d_air, d_glass, d_out,
                center=center,
                extent=1.92,
                samples=32,
                stroke_width=2,
            )
            fronts.add(front)

        self.play(FadeIn(prism), run_time=1.5)
        self.wait(0.5)

        self.add(fronts)
        self.add_subcaption("Wavefronts move at c in air until one side reaches the glass first.", duration=4)
        self.wait(4.5)

        self.add_subcaption("Inside the glass, that side moves more slowly, so the front rotates and the ray refracts.", duration=6)
        self.wait(8.0)

        self.play(FadeOut(Group(*self.mobjects)), run_time=1.2)
        self.wait(0.5)


        # ══════════════════════════════════════════════════════════════════
        # SZENE A: Reflexionsgesetz mit Kräftezerlegung
        # ══════════════════════════════════════════════════════════════════

        refl_title = Text("Reflexionsgesetz", font=MONO, font_size=36, color=WHITE)
        refl_title.to_edge(UP, buff=0.45)
        self.play(FadeIn(refl_title), run_time=0.6)

        r_hit = np.array([0.0, -0.4, 0.0])
        mirror = Line(LEFT * 5.8, RIGHT * 5.8, color=GREY_B, stroke_width=2.5)
        mirror.move_to(r_hit)
        mirror.set_opacity(0.75)

        r_normal = DashedLine(r_hit + DOWN * 0.3, r_hit + UP * 3.0,
                              dash_length=0.13, color=GREY_C, stroke_width=1.6)
        r_normal_lbl = Text("Normale n̂", font=MONO, font_size=17, color=GREY_C)
        r_normal_lbl.next_to(r_normal.get_top(), RIGHT, buff=0.12)

        self.play(Create(mirror), run_time=0.6)
        self.play(Create(r_normal), FadeIn(r_normal_lbl), run_time=0.6)
        self.wait(0.2)

        ri_deg = 40.0
        ri_rad = np.radians(ri_deg)
        d_inc = np.array([-np.sin(ri_rad), -np.cos(ri_rad), 0.0])
        ri_start = r_hit - d_inc * 2.8

        ri_ray = Arrow(ri_start, r_hit, buff=0, color=YELLOW_B,
                       stroke_width=3.5, max_tip_length_to_length_ratio=0.07)

        self.play(GrowArrow(ri_ray), run_time=0.9)
        self.wait(0.2)

        helper_normal_up = Line(r_hit, r_hit + UP * 1.5, stroke_opacity=0)
        helper_inc = Line(r_hit, ri_start, stroke_opacity=0)
        ri_arc = Angle(helper_normal_up, helper_inc, radius=0.52, color=YELLOW_B)
        ri_arc.set_stroke(width=2.4)
        ri_angle_lbl = MathTex(r"\theta_{\mathrm{ein}}", font_size=28, color=YELLOW_B)
        ri_angle_lbl.move_to(ri_arc.point_from_proportion(0.5) + LEFT * 0.38 + UP * 0.06)
        self.play(Create(ri_arc), Write(ri_angle_lbl), run_time=0.7)
        self.wait(0.3)

        n_hat = np.array([0.0, 1.0, 0.0])
        d_n = np.dot(d_inc, n_hat) * n_hat
        d_t = d_inc - d_n

        comp_origin = r_hit + UP * 0.05

        arr_inc_full = Arrow(comp_origin, comp_origin + d_inc * 1.6,
                             buff=0, color=YELLOW_B, stroke_width=2.8,
                             max_tip_length_to_length_ratio=0.1)
        arr_normal_comp = Arrow(comp_origin, comp_origin + d_n * 1.6,
                                buff=0, color=RED_B, stroke_width=2.8,
                                max_tip_length_to_length_ratio=0.1)
        arr_tang_comp = Arrow(comp_origin, comp_origin + d_t * 1.6,
                              buff=0, color=GREEN_B, stroke_width=2.8,
                              max_tip_length_to_length_ratio=0.1)

        self.play(GrowArrow(arr_tang_comp), run_time=0.7)
        self.play(GrowArrow(arr_normal_comp), run_time=0.7)
        self.wait(0.4)
        self.wait(0.5)

        d_refl = d_t - d_n
        ro_end = r_hit + d_refl * 2.8

        ro_ray = Arrow(r_hit, ro_end, buff=0, color=ORANGE,
                       stroke_width=3.5, max_tip_length_to_length_ratio=0.07)

        helper_refl = Line(r_hit, ro_end, stroke_opacity=0)
        ro_arc = Angle(helper_refl, helper_normal_up, radius=0.52, color=ORANGE)
        ro_arc.set_stroke(width=2.4)
        ro_angle_lbl = MathTex(r"\theta_{\mathrm{aus}}", font_size=28, color=ORANGE)
        ro_angle_lbl.move_to(ro_arc.point_from_proportion(0.5) + RIGHT * 0.42 + UP * 0.06)

        self.play(GrowArrow(ro_ray), run_time=0.9)
        self.play(Create(ro_arc), Write(ro_angle_lbl), run_time=0.7)
        self.wait(0.3)

        refl_formula = MathTex(r"\theta_{\mathrm{ein}} = \theta_{\mathrm{aus}}",
                               font_size=46, color=WHITE)
        refl_formula.to_corner(UR, buff=0.55).shift(DOWN * 3.5)
        self.play(Write(refl_formula), run_time=0.9)

        self.play(ri_arc.animate.set_stroke(width=5), ro_arc.animate.set_stroke(width=5),
                  run_time=0.4)
        self.play(ri_arc.animate.set_stroke(width=2.4), ro_arc.animate.set_stroke(width=2.4),
                  run_time=0.4)
        self.wait(2.0)

        refl_scene = VGroup(
            refl_title, mirror, r_normal, r_normal_lbl,
            ri_ray, ri_arc, ri_angle_lbl,
            arr_inc_full, arr_normal_comp, arr_tang_comp,
            ro_ray, ro_arc, ro_angle_lbl,
            refl_formula,
        )
        self.play(FadeOut(refl_scene), run_time=0.8)
        self.wait(0.25)

        # ══════════════════════════════════════════════════════════════════
        # SZENE B: Snell'sches Brechungsgesetz
        # ══════════════════════════════════════════════════════════════════

        snell_title = Text("Snell'sches Brechungsgesetz", font=MONO, font_size=36, color=WHITE)
        snell_title.to_edge(UP, buff=0.45)
        self.play(FadeIn(snell_title), run_time=0.6)

        s_hit = np.array([0.0, -0.2, 0.0])
        s_interface = Line(LEFT * 5.8, RIGHT * 5.8, color=GREY_B, stroke_width=2.5)
        s_interface.move_to(s_hit).set_opacity(0.75)

        lbl_m1 = MathTex(r"\mathrm{Luft}\quad n_1 = 1.00", font_size=28, color=BLUE_B)
        lbl_m1.move_to(UP * 1.7 + LEFT * 3.5)
        lbl_m2 = MathTex(r"\mathrm{Wasser}\quad n_2 = 1.33", font_size=28, color=TEAL_B)
        lbl_m2.move_to(DOWN * 1.5 + LEFT * 3.5)

        s_normal = DashedLine(s_hit + DOWN * 2.2, s_hit + UP * 2.4,
                              dash_length=0.13, color=GREY_C, stroke_width=1.6)
        s_normal_lbl = Text("Normale", font=MONO, font_size=17, color=GREY_C)
        s_normal_lbl.next_to(s_normal.get_top(), RIGHT, buff=0.12)

        self.play(Create(s_interface), FadeIn(lbl_m1), FadeIn(lbl_m2), run_time=0.8)
        self.play(Create(s_normal), FadeIn(s_normal_lbl), run_time=0.6)
        self.wait(0.2)

        s_inc_deg = 40.0
        s_inc_rad = np.radians(s_inc_deg)
        s_d_inc = np.array([np.sin(s_inc_rad), -np.cos(s_inc_rad), 0.0])
        s_inc_start = s_hit - s_d_inc * 2.6

        s_inc_ray = Arrow(s_inc_start, s_hit, buff=0, color=YELLOW_B,
                          stroke_width=3.5, max_tip_length_to_length_ratio=0.07)

        helper_s_normal_up = Line(s_hit, s_hit + UP * 1.5, stroke_opacity=0)
        helper_s_inc = Line(s_hit, s_inc_start, stroke_opacity=0)
        s1_arc = Angle(helper_s_normal_up, helper_s_inc, radius=0.5, color=YELLOW_B)
        s1_arc.set_stroke(width=2.4)
        s_theta1_lbl = MathTex(r"\theta_1", font_size=28, color=YELLOW_B)
        s_theta1_lbl.move_to(s1_arc.point_from_proportion(0.5) + RIGHT * 0.35 + UP * 0.1)

        self.play(GrowArrow(s_inc_ray), run_time=0.8)
        self.play(Create(s1_arc), Write(s_theta1_lbl), run_time=0.6)
        self.wait(0.2)

        n1, n2 = 1.0, 1.33
        sin_t2 = n1 * np.sin(s_inc_rad) / n2
        s_refr_rad = np.arcsin(sin_t2)
        s_d_refr = np.array([np.sin(s_refr_rad), -np.cos(s_refr_rad), 0.0])
        s_refr_end = s_hit + s_d_refr * 2.5

        s_refr_ray = Arrow(s_hit, s_refr_end, buff=0, color=TEAL_B,
                           stroke_width=3.5, max_tip_length_to_length_ratio=0.07)

        helper_s_normal_down = Line(s_hit, s_hit + DOWN * 1.5, stroke_opacity=0)
        helper_s_refr = Line(s_hit, s_refr_end, stroke_opacity=0)
        s2_arc = Angle(helper_s_normal_down, helper_s_refr, radius=0.5, color=TEAL_B)
        s2_arc.set_stroke(width=2.4)
        s_theta2_lbl = MathTex(r"\theta_2", font_size=28, color=TEAL_B)
        s_theta2_lbl.move_to(s2_arc.point_from_proportion(0.5) + RIGHT * 0.34 + DOWN * 0.12)

        self.play(GrowArrow(s_refr_ray), run_time=0.8)
        self.play(Create(s2_arc), Write(s_theta2_lbl), run_time=0.6)
        self.wait(0.3)

        snell_formula = MathTex(r"n_1 \sin\theta_1 = n_2 \sin\theta_2",
                                font_size=46, color=WHITE)
        snell_formula.to_corner(UR, buff=0.6).shift(DOWN * 0.5)
        self.play(Write(snell_formula), run_time=1.1)
        self.wait(0.4)

        self.wait(2.5)

        snell_scene = VGroup(
            snell_title, s_interface, lbl_m1, lbl_m2,
            s_normal, s_normal_lbl,
            s_inc_ray, s1_arc, s_theta1_lbl,
            s_refr_ray, s2_arc, s_theta2_lbl,
            snell_formula,
        )
        self.play(FadeOut(snell_scene), run_time=0.8)
        self.wait(0.3)

        # ── 2. Wassertropfen-Szene ─────────────────────────────────────────
        droplet = build_droplet()
        self.add(droplet)

        DEMO_WAVELENGTH = 540.0
        wavelength = ValueTracker(DEMO_WAVELENGTH)
        y_offset = ValueTracker(0.95)

        beam = always_redraw(lambda: build_beam_group(wavelength.get_value(), y_offset.get_value()))
        beam.set_z_index(6)

        wavelength_caption = Text(
            f"λ = {int(DEMO_WAVELENGTH)} nm",
            font=MONO, font_size=26, color=WHITE,
        ).to_corner(UL, buff=0.55).set_z_index(8)

        self.play(FadeIn(beam), FadeIn(wavelength_caption), run_time=1.4)
        self.wait(0.8)

        angle_marker = always_redraw(
            lambda: build_angle_marker(wavelength.get_value(), y_offset.get_value())
        )
        self.add(angle_marker)

        for color_name, wavelength_nm in SELECTED_COLORS:

            angle_readout = always_redraw(
                lambda w=wavelength_nm: Text(
                    f"{(trace_ray_bundle(w, y_offset.get_value())['primary']['angle_deg']):.2f}°"
                    if trace_ray_bundle(w, y_offset.get_value())["primary"] is not None
                    else "—",
                    font=MONO,
                    font_size=28,
                    color=WHITE,
                )
                .to_corner(UR, buff=0.55)
                .set_z_index(8)
            )

            color_label = Text(
                f"{color_name}  λ = {int(wavelength_nm)} nm",
                font=MONO, font_size=24, color=WHITE,
            ).to_corner(UL, buff=0.55).shift(DOWN * 0.55)

            self.play(
                wavelength.animate.set_value(wavelength_nm),
                y_offset.animate.set_value(0.05),
                FadeIn(angle_readout),
                FadeIn(color_label),
                run_time=1.2,
            )
            self.wait(0.3)
            self.play(y_offset.animate.set_value(Y_MAX), run_time=6.5, rate_func=linear)
            self.wait(0.5)

            self.play(FadeOut(angle_readout), FadeOut(color_label), run_time=0.5)

        self.play(
            FadeOut(beam),
            FadeOut(wavelength_caption),
            FadeOut(angle_marker),
            FadeOut(droplet),
            run_time=1.0,
        )
        self.wait(0.3)

        for color_name, wavelength_nm in SELECTED_COLORS:
            accent = wavelength_to_rgb(wavelength_nm)
            graph = build_final_graph(color_name, wavelength_nm, accent)
            graph.center()

            self.play(FadeIn(graph), run_time=1.0)
            self.wait(2.8)
            self.play(FadeOut(graph), run_time=0.8)
            self.wait(0.3)

        all_samples = []
        for color_name, wavelength_nm in SELECTED_COLORS:
            all_samples.append((color_name, wavelength_nm, compute_angle_curve(wavelength_nm)))

        all_y = [y for _, _, pts in all_samples for _, y in pts]
        y_min_global = np.floor(min(all_y) - 0.5)
        y_max_global = np.ceil(max(all_y) + 0.5)
        y_tick_cmp = round((y_max_global - y_min_global) / 4)
        if y_tick_cmp < 1:
            y_tick_cmp = 1

        cmp_axes = Axes(
            x_range=[0, Y_MAX, round(Y_MAX / 4, 2)],
            y_range=[y_min_global, y_max_global, y_tick_cmp],
            x_length=9.5,
            y_length=5.5,
            axis_config={
                "color": GREY_B,
                "stroke_opacity": 0.7,
                "stroke_width": 2.0,
                "include_numbers": False,
            },
            tips=False,
        )
        cmp_axes.center()

        cmp_x_label = Text("Einfallshöhe y [m]", font=MONO, font_size=24, color=GREY_A)
        cmp_x_label.next_to(cmp_axes, DOWN, buff=0.45)

        cmp_y_label = Text("Ablenkwinkel [°]", font=MONO, font_size=24, color=GREY_A)
        cmp_y_label.rotate(PI / 2)
        cmp_y_label.next_to(cmp_axes, LEFT, buff=0.55)

        cmp_title = Text("Dispersionsvergleich", font=MONO, font_size=32, color=WHITE)
        cmp_title.next_to(cmp_axes, UP, buff=0.45)

        legend_items = VGroup()
        curves = VGroup()
        for color_name, wavelength_nm, pts in all_samples:
            accent = wavelength_to_rgb(wavelength_nm)
            curve_pts = [cmp_axes.c2p(x, y) for x, y in pts]
            c = VMobject(color=accent)
            c.set_points_as_corners(curve_pts)
            c.set_stroke(color=accent, width=5.0)
            curves.add(c)

            leg = Text(
                f"{color_name}  {int(wavelength_nm)} nm",
                font=MONO, font_size=22, color=accent,
            )
            legend_items.add(leg)

        legend_items.arrange(DOWN, aligned_edge=LEFT, buff=0.28)
        legend_items.next_to(cmp_axes, RIGHT, buff=0.55)

        cmp_group = VGroup(cmp_axes, cmp_x_label, cmp_y_label, cmp_title, curves, legend_items)

        self.play(FadeIn(cmp_axes), FadeIn(cmp_x_label), FadeIn(cmp_y_label), FadeIn(cmp_title), run_time=1.0)
        for c in curves:
            self.play(Create(c), run_time=1.6, rate_func=linear)
        self.play(FadeIn(legend_items), run_time=0.7)
        self.wait(3.5)

        self.play(FadeOut(cmp_group), run_time=1.0)
        self.wait(0.3)


class RainbowPresentation(IntroBaseScene):
    def construct(self):
        # ══════════════════════════════════════════════════════════════════
        # TITEL / ORIENTIERUNG
        # ══════════════════════════════════════════════════════════════════
        title = Text("Der Regenbogen", font=MONO, font_size=64, color=WHITE, weight=BOLD)
        title.move_to(UP * 0.55)

        self.play(FadeIn(title, shift=UP * 0.2), run_time=1.0)
        self.wait(0.4)
        self.play(title.animate.scale(0.56).to_corner(UL, buff=0.55), run_time=0.9)

        title_accent = Line(ORIGIN, RIGHT * 2.7, color=INTRO_PRIMARY, stroke_width=4)
        title_accent.next_to(title, DOWN, aligned_edge=LEFT, buff=0.16)

        roadmap = VGroup(
            Text("1. Was Brechung mikroskopisch verursacht", font=MONO, font_size=22, color=INTRO_PRIMARY),
            Text("2. Wie Reflexion und Brechung geometrisch beschrieben wird", font=MONO, font_size=22, color=YELLOW_B),
            Text("3. Wie im Wassertropfen der Regenbogen entsteht", font=MONO, font_size=22, color=BLUE_B),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.32)
        roadmap.next_to(title_accent, DOWN, aligned_edge=LEFT, buff=0.42)

        self.play(Create(title_accent), run_time=0.5)
        self.play(LaggedStart(*[Write(m) for m in roadmap], lag_ratio=0.18), run_time=1.8)
        self.wait(2.0)
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.7)
        self.wait(0.2)

        # ══════════════════════════════════════════════════════════════════
        # REFLEXION IM TROPFEN
        # ══════════════════════════════════════════════════════════════════
        refl_title = intro_make_title("Reflexion: Warum der Strahl im Tropfen umgelenkt wird")
        self.play(Write(refl_title), run_time=0.8)
        note = None
        note = self.change_note(note, "Im Regentropfen reicht eine Brechung allein nicht aus. Der Strahl wird zusätzlich an der Rückseite reflektiert.")

        r_hit = np.array([0.0, -0.4, 0.0])
        mirror = Line(LEFT * 5.8, RIGHT * 5.8, color=GREY_B, stroke_width=2.5).move_to(r_hit)
        mirror.set_opacity(0.75)
        r_normal = DashedLine(r_hit + DOWN * 0.3, r_hit + UP * 3.0, dash_length=0.13, color=GREY_C, stroke_width=1.6)
        r_normal_lbl = Text("Normale", font=MONO, font_size=17, color=GREY_C).next_to(r_normal.get_top(), RIGHT, buff=0.12)
        self.play(Create(mirror), Create(r_normal), Write(r_normal_lbl), run_time=0.9)

        ri_deg = 40.0
        ri_rad = np.radians(ri_deg)
        d_inc = np.array([-np.sin(ri_rad), -np.cos(ri_rad), 0.0])
        ri_start = r_hit - d_inc * 2.8
        ri_ray = Arrow(ri_start, r_hit, buff=0, color=YELLOW_B, stroke_width=3.5, max_tip_length_to_length_ratio=0.07)
        self.play(GrowArrow(ri_ray), run_time=0.8)

        helper_normal_up = Line(r_hit, r_hit + UP * 1.5, stroke_opacity=0)
        helper_inc = Line(r_hit, ri_start, stroke_opacity=0)
        ri_arc = Angle(helper_normal_up, helper_inc, radius=0.52, quadrant=(1, 1), color=YELLOW_B)
        ri_arc.set_stroke(width=2.4)
        ri_angle_lbl = MathTex(r"\theta_{\mathrm{ein}}", font_size=28, color=YELLOW_B)
        ri_angle_lbl.move_to(ri_arc.point_from_proportion(0.5) + LEFT * 0.38 + UP * 0.06)
        self.play(Create(ri_arc), Write(ri_angle_lbl), run_time=0.6)

        n_hat = np.array([0.0, 1.0, 0.0])
        d_n = np.dot(d_inc, n_hat) * n_hat
        d_t = d_inc - d_n

        d_refl = d_t - d_n
        ro_end = r_hit + d_refl * 2.8
        ro_ray = Arrow(r_hit, ro_end, buff=0, color=ORANGE, stroke_width=3.5, max_tip_length_to_length_ratio=0.07)
        helper_refl = Line(r_hit, ro_end, stroke_opacity=0)
        ro_arc = Angle(helper_refl, helper_normal_up, radius=0.52, quadrant=(-1, 1), color=ORANGE)
        ro_arc.set_stroke(width=2.4)
        ro_angle_lbl = MathTex(r"\theta_{\mathrm{aus}}", font_size=28, color=ORANGE)
        ro_angle_lbl.move_to(ro_arc.point_from_proportion(0.5) + RIGHT * 0.42 + UP * 0.06)
        self.play(GrowArrow(ro_ray), Create(ro_arc), Write(ro_angle_lbl), run_time=1.0)

        refl_formula = MathTex(r"\theta_{\mathrm{ein}} = \theta_{\mathrm{aus}}", font_size=38, color=WHITE).to_corner(UR, buff=0.55).shift(DOWN * 3.5)
        self.play(Write(refl_formula), run_time=0.8)
        self.wait(1.5)
        self.clean_end()

        # ══════════════════════════════════════════════════════════════════
        # SNELLIUS
        # ══════════════════════════════════════════════════════════════════
        snell_title = intro_make_title("Brechungsgesetz nach Snellius")
        self.play(Write(snell_title), run_time=0.8)
        note = None
        note = self.change_note(note, "Für die geometrische Beschreibung der Brechung verwendet man das Snelliussche Gesetz.")

        s_hit = np.array([0.0, -0.2, 0.0])
        s_interface = Line(LEFT * 5.8, RIGHT * 5.8, color=GREY_B, stroke_width=2.5).move_to(s_hit).set_opacity(0.75)
        lbl_m1 = MathTex(r"\mathrm{Luft}\quad n_1 = 1.00", font_size=28, color=BLUE_B).move_to(UP * 1.7 + LEFT * 3.5)
        lbl_m2 = MathTex(r"\mathrm{Wasser}\quad n_2 = 1.33", font_size=28, color=TEAL_B).move_to(DOWN * 1.5 + LEFT * 3.5)
        s_normal = DashedLine(s_hit + DOWN * 2.2, s_hit + UP * 2.4, dash_length=0.13, color=GREY_C, stroke_width=1.6)
        s_normal_lbl = Text("Normale", font=MONO, font_size=17, color=GREY_C).next_to(s_normal.get_top(), RIGHT, buff=0.12)
        self.play(Create(s_interface), Write(lbl_m1), Write(lbl_m2), Create(s_normal), Write(s_normal_lbl), run_time=1.1)

        s_inc_deg = 40.0
        s_inc_rad = np.radians(s_inc_deg)
        s_d_inc = np.array([np.sin(s_inc_rad), -np.cos(s_inc_rad), 0.0])
        s_inc_start = s_hit - s_d_inc * 2.6
        s_inc_ray = Arrow(s_inc_start, s_hit, buff=0, color=YELLOW_B, stroke_width=3.5, max_tip_length_to_length_ratio=0.07)
        helper_s_normal_up = Line(s_hit, s_hit + UP * 1.5, stroke_opacity=0)
        helper_s_inc = Line(s_hit, s_inc_start, stroke_opacity=0)
        s1_arc = Angle(helper_s_normal_up, helper_s_inc, radius=0.5, quadrant=(1, 1), color=YELLOW_B)
        s1_arc.set_stroke(width=2.4)
        s_theta1_lbl = MathTex(r"\theta_1", font_size=28, color=YELLOW_B)
        s_theta1_lbl.move_to(s1_arc.point_from_proportion(0.5) + RIGHT * 0.35 + UP * 0.1)
        self.play(GrowArrow(s_inc_ray), Create(s1_arc), Write(s_theta1_lbl), run_time=0.9)

        n1, n2 = 1.0, 1.33
        sin_t2 = n1 * np.sin(s_inc_rad) / n2
        s_refr_rad = np.arcsin(sin_t2)
        s_d_refr = np.array([np.sin(s_refr_rad), -np.cos(s_refr_rad), 0.0])
        s_refr_end = s_hit + s_d_refr * 2.5
        s_refr_ray = Arrow(s_hit, s_refr_end, buff=0, color=TEAL_B, stroke_width=3.5, max_tip_length_to_length_ratio=0.07)
        helper_s_normal_down = Line(s_hit, s_hit + DOWN * 1.5, stroke_opacity=0)
        helper_s_refr = Line(s_hit, s_refr_end, stroke_opacity=0)
        s2_arc = Angle(helper_s_normal_down, helper_s_refr, radius=0.5, quadrant=(1, -1), color=TEAL_B)
        s2_arc.set_stroke(width=2.4)
        s_theta2_lbl = MathTex(r"\theta_2", font_size=28, color=TEAL_B)
        s_theta2_lbl.move_to(s2_arc.point_from_proportion(0.5) + RIGHT * 0.34 + DOWN * 0.12)
        self.play(GrowArrow(s_refr_ray), Create(s2_arc), Write(s_theta2_lbl), run_time=0.9)

        snell_formula = MathTex(r"n_1 \sin(\theta_1) = n_2 \sin(\theta_2)", font_size=38, color=WHITE).to_corner(UR, buff=0.6).shift(DOWN * 0.6)
        self.play(Write(snell_formula), run_time=0.9)
        self.wait(1.7)
        self.clean_end()

        # ══════════════════════════════════════════════════════════════════
        # c, LICHTGESCHWINDIGKEIT IN WASSER, BRECHUNGSINDEX
        # ══════════════════════════════════════════════════════════════════
        basics_title = Text("Grundbegriffe: c, Geschwindigkeit und n", font=MONO, font_size=40, color=WHITE)
        basics_title.to_edge(UP, buff=0.85)

        left_panel = RoundedRectangle(width=5.3, height=4.3, corner_radius=0.16, stroke_color=INTRO_STRUCT, stroke_opacity=0.4)
        right_panel = left_panel.copy()
        left_panel.shift(LEFT * 3.35 + DOWN * 0.2)
        right_panel.shift(RIGHT * 3.35 + DOWN * 0.2)

        c_head = Text("Im Vakuum", font=MONO, font_size=24, color=INTRO_PRIMARY)
        c_head.next_to(left_panel, UP, buff=0.16)
        c_lines = VGroup(
            Text("c = Lichtgeschwindigkeit im Vakuum", font=MONO, font_size=18, color=INTRO_WHITEISH),
            MathTex(r"c = 299\,792\,458\,\mathrm{m/s}", font_size=34, color=INTRO_PRIMARY),
            Text("Das ist der Referenzwert.", font=MONO, font_size=18, color=GREY_A),
            Text("Alle Brechungsindizes vergleichen", font=MONO, font_size=18, color=GREY_A),
            Text("ein Material mit genau diesem Wert.", font=MONO, font_size=18, color=GREY_A),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.16)
        c_lines.move_to(left_panel.get_center())

        n_head = Text("In Wasser", font=MONO, font_size=24, color=TEAL_B)
        n_head.next_to(right_panel, UP, buff=0.16)
        water_lines = VGroup(
            MathTex(r"n \approx 1.333", font_size=34, color=TEAL_B),
            MathTex(r"n = \frac{c}{v}", font_size=40, color=YELLOW_B),
            MathTex(r"v = \frac{c}{n}", font_size=36, color=YELLOW_B),
            MathTex(r"v \approx 2.25 \cdot 10^8\,\mathrm{m/s}", font_size=34, color=INTRO_ACCENT),
            MathTex(r"v \approx 0.75c", font_size=34, color=INTRO_ACCENT),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
        water_lines.move_to(right_panel.get_center())

        note = None
        note = self.change_note(note, "c ist die Lichtgeschwindigkeit im Vakuum. Der Brechungsindex wird über n = c / v definiert.")
        self.play(Write(basics_title), run_time=0.9)
        self.play(FadeIn(left_panel), Write(c_head), LaggedStart(*[Write(m) for m in c_lines], lag_ratio=0.12), run_time=1.4)
        self.play(FadeIn(right_panel), Write(n_head), LaggedStart(*[Write(m) for m in water_lines], lag_ratio=0.12), run_time=1.4)
        note = self.change_note(note, "Für Wasser mit n ungefähr 1.333 folgt v = c / n. Das ergibt rund 2.25 mal 10 hoch 8 Meter pro Sekunde, also etwa 0.75c.")
        self.wait(2.4)
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.7)
        self.wait(0.2)

        # ══════════════════════════════════════════════════════════════════
        # MODELLE DES LICHTS
        # ══════════════════════════════════════════════════════════════════
        intro_title = Text("Modelle des Lichts", font=MONO, font_size=42, color=WHITE)
        intro_title.to_edge(UP, buff=1.0)

        model_wave = Text("1. Wellenmodell (Wellenfronten)", font=MONO, font_size=28, color=PRISM_AIR_COLOR)
        model_ray = Text("2. Teilchenmodell (Lichtstrahlen)", font=MONO, font_size=28, color=YELLOW_B)
        models = VGroup(model_wave, model_ray).arrange(DOWN, aligned_edge=LEFT, buff=1.4)
        models.next_to(intro_title, DOWN, buff=1.0)

        self.play(Write(intro_title), run_time=0.9)
        self.play(Write(model_wave), run_time=0.8)
        self.play(Write(model_ray), run_time=0.8)
        self.wait(2.0)
        self.play(FadeOut(intro_title), FadeOut(models), run_time=0.7)
        self.wait(0.2)

        # ══════════════════════════════════════════════════════════════════
        # INTRO 1
        # ══════════════════════════════════════════════════════════════════
        title = intro_make_title("Phasenverschiebung: Was passiert im Material?")
        self.play(Write(title), run_time=0.8)

        left_box = RoundedRectangle(width=5.8, height=3.7, corner_radius=0.18,
                                    stroke_color=INTRO_STRUCT, stroke_opacity=0.45)
        right_box = left_box.copy()
        left_box.shift(LEFT * 3.45 + UP * 0.15)
        right_box.shift(RIGHT * 3.45 + UP * 0.15)
        left_label = intro_section_label("Vakuum", INTRO_PRIMARY).next_to(left_box, UP, buff=0.16)
        right_label = intro_section_label("Material", INTRO_SECONDARY).next_to(right_box, UP, buff=0.16)

        ax_l = Axes(x_range=[0, 6.2, 1], y_range=[-1.5, 1.5, 1], x_length=4.8, y_length=2.1,
                    axis_config={"color": INTRO_STRUCT, "stroke_opacity": 0.25, "include_tip": False})
        ax_r = ax_l.copy()
        ax_l.move_to(left_box)
        ax_r.move_to(right_box)

        phase = ValueTracker(0)
        wave_l = always_redraw(lambda: ax_l.plot(lambda x: 0.75 * np.sin(2.0 * x - phase.get_value()), color=INTRO_PRIMARY, stroke_width=4))
        wave_r = always_redraw(lambda: ax_r.plot(lambda x: 0.75 * np.sin(2.4 * x - 0.78 * phase.get_value() - 0.8), color=INTRO_ACCENT, stroke_width=4))

        freq_text = Text("gleiche Lichtfrequenz", font=MONO, font_size=16, color=INTRO_WHITEISH)
        freq_text.move_to(DOWN * 1.55)
        caution = Text("Nicht: Stop-and-Go", font=MONO, font_size=INTRO_SMALL_SIZE, color=INTRO_WARNING)
        caution.next_to(freq_text, DOWN, buff=0.16)

        note = None
        note = self.change_note(note, "Frage: Warum ist die Welle im Material phasenverschoben und scheinbar langsamer?")
        self.play(FadeIn(left_box), FadeIn(right_box), Write(left_label), Write(right_label), FadeIn(ax_l), FadeIn(ax_r), run_time=1.0)
        self.play(Create(wave_l), Create(wave_r), Write(freq_text), run_time=1.4)
        self.play(phase.animate.set_value(4 * PI), run_time=3.6, rate_func=linear)
        note = self.change_note(note, "Wichtig: Das Licht wird nicht wie ein Ball immer wieder angehalten. Die Ursache ist die Antwort der Ladungen im Material.")
        self.play(Write(caution), run_time=0.7)
        self.wait(0.8)
        self.clean_end()

        # ══════════════════════════════════════════════════════════════════
        # INTRO 2
        # ══════════════════════════════════════════════════════════════════
        title = intro_make_title("1) Das Lichtfeld treibt ein gebundenes Elektron an")
        self.play(Write(title), run_time=0.8)
        note = None
        note = self.change_note(note, "Eine Lichtwelle besitzt ein elektrisches Feld. Dieses Feld übt auf geladene Teilchen eine Kraft aus.")

        atom_center = UP * 0.25
        orbit_line = Line(atom_center + LEFT * 2.2, atom_center + RIGHT * 2.2, color=INTRO_STRUCT, stroke_opacity=0.25)
        nucleus = Dot(atom_center, radius=0.18, color=INTRO_WARNING)
        nucleus_label = Text("Atomkern", font=MONO, font_size=INTRO_SMALL_SIZE, color=INTRO_WHITEISH).next_to(nucleus, UP, buff=0.18)
        guide = DashedLine(atom_center + LEFT * 2.0, atom_center + RIGHT * 2.0, color=INTRO_STRUCT, stroke_opacity=0.25)

        t = ValueTracker(0)
        response = lambda: 1.15 * np.sin(t.get_value() - 0.85)

        field_arrow = always_redraw(lambda: Arrow(atom_center + LEFT * 4.8, atom_center + LEFT * 4.8 + RIGHT * (1.2 + 0.9 * np.sin(t.get_value())),
                                                   buff=0, stroke_width=6, max_stroke_width_to_length_ratio=10,
                                                   max_tip_length_to_length_ratio=0.2, color=INTRO_PRIMARY))
        field_label = Text("E-Feld der Lichtwelle", font=MONO, font_size=INTRO_SMALL_SIZE, color=INTRO_PRIMARY)
        field_label.next_to(atom_center + LEFT * 4.0 + UP * 1.3, DOWN, buff=0.1)

        spring = always_redraw(lambda: ParametricFunction(
            lambda u: np.array([
                atom_center[0] + (-0.2 + (response() + 0.2) * u),
                atom_center[1] + 0.18 * np.sin(10 * PI * u),
                0,
            ]),
            t_range=[0, 1], color=INTRO_STRUCT, stroke_width=2.5, stroke_opacity=0.6,
        ))
        electron = always_redraw(lambda: Dot(atom_center + RIGHT * response(), radius=0.14, color=INTRO_SECONDARY))
        electron_label = always_redraw(lambda: Text("Elektron", font=MONO, font_size=INTRO_SMALL_SIZE, color=INTRO_SECONDARY).next_to(electron, DOWN, buff=0.16))
        rest_marker = Dot(atom_center + LEFT * 0.05, radius=0.05, color=INTRO_STRUCT)

        self.play(FadeIn(orbit_line), FadeIn(guide), FadeIn(nucleus), Write(nucleus_label), FadeIn(rest_marker), run_time=0.9)
        self.play(FadeIn(field_arrow), Write(field_label), FadeIn(spring), FadeIn(electron), FadeIn(electron_label), run_time=0.9)
        self.play(t.animate.set_value(3 * PI), run_time=3.6, rate_func=linear)
        note = self.change_note(note, "Das Elektron folgt nicht exakt sofort. Es ist an das Atom gebunden und hat Trägheit. Deshalb hinkt seine Bewegung dem Feld etwas hinterher.")
        self.play(t.animate.set_value(6 * PI), run_time=3.2, rate_func=linear)
        self.wait(0.8)
        self.clean_end()

        # ══════════════════════════════════════════════════════════════════
        # INTRO 3
        # ══════════════════════════════════════════════════════════════════
        title = intro_make_title("2) Wann sendet ein Teilchen elektromagnetische Wellen aus?")
        self.play(Write(title), run_time=0.8)
        note = None
        note = self.change_note(note, "Die kurze Antwort lautet: Wenn eine Ladung beschleunigt wird. Reine Ruhe oder gleichförmige Bewegung strahlen nicht ab.")

        panels = VGroup(*[
            RoundedRectangle(width=3.95, height=4.0, corner_radius=0.16, stroke_color=INTRO_STRUCT, stroke_opacity=0.45)
            for _ in range(3)
        ]).arrange(RIGHT, buff=0.35).shift(UP * 0.45)
        labels = VGroup(
            Text("Ruhe", font=MONO, font_size=INTRO_SMALL_SIZE, color=INTRO_WHITEISH),
            Text("gleichförmig", font=MONO, font_size=INTRO_SMALL_SIZE, color=INTRO_WHITEISH),
            Text("beschleunigt", font=MONO, font_size=INTRO_SMALL_SIZE, color=INTRO_WHITEISH),
        )
        for lbl, panel in zip(labels, panels):
            lbl.next_to(panel, UP, buff=0.12)
        self.play(FadeIn(panels), FadeIn(labels), run_time=0.9)

        p1 = panels[0].get_center()
        charge1 = Dot(p1, radius=0.12, color=INTRO_SECONDARY)
        halo1 = Circle(radius=0.55, color=INTRO_PRIMARY, stroke_opacity=0.18).move_to(p1)
        self.play(FadeIn(charge1), FadeIn(halo1), run_time=0.8)

        p2 = panels[1].get_center()
        charge2 = Dot(p2 + LEFT * 1.0, radius=0.12, color=INTRO_SECONDARY)
        arrow2 = Arrow(p2 + LEFT * 1.35, p2 + RIGHT * 1.2, buff=0, color=INTRO_PRIMARY, stroke_width=5)
        trail2 = VGroup(*[Circle(radius=r, color=INTRO_PRIMARY, stroke_opacity=0.08).move_to(p2 + RIGHT * (0.4 * r)) for r in [0.35, 0.7, 1.05]])
        self.play(FadeIn(charge2), GrowArrow(arrow2), FadeIn(trail2), run_time=0.8)

        p3 = panels[2].get_center()
        osc = ValueTracker(0)
        charge3 = always_redraw(lambda: Dot(p3 + RIGHT * (0.8 * np.sin(osc.get_value())), radius=0.12, color=INTRO_SECONDARY))
        path3 = Line(p3 + LEFT * 1.1, p3 + RIGHT * 1.1, color=INTRO_STRUCT, stroke_opacity=0.25)
        ripples = VGroup(*[Circle(radius=0.15, color=INTRO_ACCENT, stroke_opacity=0.55).move_to(p3) for _ in range(3)])
        self.play(FadeIn(path3), FadeIn(charge3), run_time=0.8)
        self.play(osc.animate.set_value(4 * PI), *[r.animate.scale(8).set_stroke(opacity=0) for r in ripples], run_time=2.8, rate_func=linear)
        note = self.change_note(note, "Eine schwingende Ladung wird fortwährend beschleunigt. Genau diese zeitlich veränderliche Bewegung erzeugt elektromagnetische Strahlung.")
        self.wait(0.8)
        self.clean_end()

        # ══════════════════════════════════════════════════════════════════
        # INTRO 4
        # ══════════════════════════════════════════════════════════════════
        title = intro_make_title("3) Viele Atome zusammen verschieben die Phase")
        self.play(Write(title), run_time=0.8)
        note = None
        note = self.change_note(note, "In einem Material schwingen sehr viele gebundene Elektronen. Ihre Antwort überlagert sich mit der einfallenden Welle.")

        material_box = RoundedRectangle(width=9.6, height=3.1, corner_radius=0.15,
                                        stroke_color=INTRO_STRUCT, stroke_opacity=0.45,
                                        fill_color="#131A22", fill_opacity=0.35).shift(UP * 0.3)
        material_label = Text("Material", font=MONO, font_size=INTRO_SMALL_SIZE, color=INTRO_SECONDARY).next_to(material_box, UP, buff=0.12)
        atoms = VGroup()
        for x in np.linspace(-4.0, 4.0, 7):
            core = Dot(np.array([x, 0.1, 0]), radius=0.08, color=INTRO_WARNING)
            electron = Dot(np.array([x + 0.22, 0.1, 0]), radius=0.05, color=INTRO_SECONDARY)
            bond = Line(core.get_center(), electron.get_center(), color=INTRO_STRUCT, stroke_opacity=0.35)
            atoms.add(VGroup(core, bond, electron))
        atoms.shift(UP * 0.3)

        ax_top = Axes(x_range=[0, 10, 1], y_range=[-1.4, 1.4, 1], x_length=10.0, y_length=1.4,
                      axis_config={"color": INTRO_STRUCT, "stroke_opacity": 0.18, "include_tip": False})
        ax_mid = ax_top.copy()
        ax_bot = ax_top.copy()
        ax_top.move_to(UP * 2.15)
        ax_mid.move_to(UP * 0.0)
        ax_bot.move_to(DOWN * 1.75)

        incoming_label = Text("einfallende Welle", font=MONO, font_size=INTRO_SMALL_SIZE, color=INTRO_PRIMARY)
        incoming_label.move_to(ax_top.get_left() + RIGHT * 1.55 + UP * 0.55)
        response_label = Text("Materialantwort", font=MONO, font_size=INTRO_SMALL_SIZE, color=INTRO_SECONDARY)
        response_label.move_to(ax_mid.get_left() + RIGHT * 1.6 + UP * 0.55)
        result_label = Text("resultierende Welle", font=MONO, font_size=INTRO_SMALL_SIZE, color=INTRO_ACCENT)
        result_label.move_to(ax_bot.get_left() + RIGHT * 1.75 + UP * 0.55)

        phase = ValueTracker(0)
        incoming = always_redraw(lambda: ax_top.plot(lambda x: 0.55 * np.sin(1.7 * x - phase.get_value()), color=INTRO_PRIMARY, stroke_width=4))
        response = always_redraw(lambda: ax_mid.plot(lambda x: 0.38 * np.sin(1.7 * x - phase.get_value() - 0.95), color=INTRO_SECONDARY, stroke_width=4))
        result = always_redraw(lambda: ax_bot.plot(lambda x: 0.55 * np.sin(1.7 * x - phase.get_value() - 0.42), color=INTRO_ACCENT, stroke_width=4))

        self.play(FadeIn(material_box), Write(material_label), FadeIn(atoms), run_time=0.9)
        self.play(FadeIn(ax_top), FadeIn(ax_mid), FadeIn(ax_bot), Write(incoming_label), Write(response_label), Write(result_label), run_time=0.9)
        self.play(Create(incoming), Create(response), Create(result), run_time=1.2)
        self.play(phase.animate.set_value(4 * PI), run_time=3.4, rate_func=linear)

        shift_arrow = DoubleArrow(ax_bot.c2p(4.4, 0.95), ax_bot.c2p(5.0, 0.95), color=INTRO_ACCENT, stroke_width=4)
        shift_text = Text("Phasenverschiebung", font=MONO, font_size=INTRO_SMALL_SIZE, color=INTRO_ACCENT).next_to(shift_arrow, UP, buff=0.12)
        self.play(FadeIn(shift_arrow), Write(shift_text), run_time=0.7)
        note = self.change_note(note, "Die vom Material erzeugten Felder addieren sich zur ursprünglichen Welle. Dadurch entsteht eine neue Welle mit verschobener Phase.")
        self.play(phase.animate.set_value(7 * PI), run_time=2.8, rate_func=linear)
        self.wait(0.8)
        self.clean_end()

        # ══════════════════════════════════════════════════════════════════
        # INTRO 5
        # ══════════════════════════════════════════════════════════════════
        title = intro_make_title("4) An der Grenzfläche wird daraus Brechung")
        self.play(Write(title), run_time=0.8)
        note = None
        note = self.change_note(note, "Trifft eine Wellenfront schräg auf Wasser, gelangt ihr unterer Teil zuerst in das Medium und wird dort zuerst langsamer.")

        boundary = Line(LEFT * 6.3 + DOWN * 0.45, RIGHT * 6.3 + DOWN * 0.45, color=INTRO_WHITEISH, stroke_width=3)
        air = Text("Luft", font=MONO, font_size=INTRO_SMALL_SIZE, color=INTRO_PRIMARY).move_to(LEFT * 5.4 + UP * 2.5)
        water = Text("Wasser", font=MONO, font_size=INTRO_SMALL_SIZE, color=INTRO_SECONDARY).move_to(LEFT * 5.2 + DOWN * 2.5)
        self.play(Create(boundary), Write(air), Write(water), run_time=0.9)

        fronts_top = VGroup(*[
            Line(LEFT * 1.8 + UP * (2.8 - 0.7 * i), RIGHT * 1.8 + UP * (1.7 - 0.7 * i), color=INTRO_PRIMARY, stroke_width=3)
            for i in range(4)
        ]).shift(LEFT * 1.8)
        beam = Arrow(LEFT * 4.4 + UP * 2.3, LEFT * 1.9 + UP * 0.8, buff=0, color=INTRO_PRIMARY, stroke_width=5)
        self.play(LaggedStart(*[Create(f) for f in fronts_top], lag_ratio=0.15), GrowArrow(beam), run_time=1.4)

        note = self.change_note(note, "Weil ein Teil der Wellenfront früher im Wasser ist als der andere, dreht sich die ganze Front. Genau diese Drehung ändert die Ausbreitungsrichtung.")
        fronts_bottom = VGroup(*[
            Line(LEFT * 1.0 + DOWN * (0.4 + 0.6 * i), RIGHT * 1.5 + DOWN * (0.9 + 0.6 * i), color=INTRO_ACCENT, stroke_width=3)
            for i in range(4)
        ]).shift(RIGHT * 1.05)
        beam2 = Arrow(LEFT * 1.2 + DOWN * 0.1, RIGHT * 1.0 + DOWN * 1.7, buff=0, color=INTRO_ACCENT, stroke_width=5)
        normal = DashedLine(ORIGIN + UP * 2.2 + RIGHT * 0.2, ORIGIN + DOWN * 2.2 + RIGHT * 0.2, color=INTRO_STRUCT, stroke_opacity=0.4)
        self.play(Create(normal), LaggedStart(*[Create(f) for f in fronts_bottom], lag_ratio=0.15), GrowArrow(beam2), run_time=1.6)

        top_dot = Dot(LEFT * 1.8 + UP * 0.75, radius=0.08, color=INTRO_PRIMARY)
        bot_dot = Dot(LEFT * 0.6 + DOWN * 0.8, radius=0.08, color=INTRO_ACCENT)
        self.play(FadeIn(top_dot), FadeIn(bot_dot), run_time=0.7)
        self.wait(0.8)
        self.clean_end()

        # ══════════════════════════════════════════════════════════════════
        # PRISMA / WELLENFRONTEN
        # ══════════════════════════════════════════════════════════════════
        wave_title = intro_make_title("Wellenbild der Brechung am Prisma")
        self.play(Write(wave_title), run_time=0.8)
        note = None
        note = self.change_note(note, "Im Wellenmodell sieht man direkt, warum sich die Ausbreitungsrichtung ändert: Eine Seite der Wellenfront wird zuerst verzögert.")

        p1 = np.array([-10.0, -12.0, 0.0])
        p2 = np.array([10.0, 22.64, 0.0])
        p3 = np.array([30.0, -12.0, 0.0])
        prism_center = (p1 + p2 + p3) / 3
        entry_normal = outward_normal(p1, p2, prism_center)
        exit_normal = outward_normal(p2, p3, prism_center)
        d_air = np.array([1.0, 0.0, 0.0])
        d_glass = refract_prism(d_air, entry_normal, 1.0, 1.5)
        d_out = refract_prism(d_glass, exit_normal, 1.5, 1.0)

        prism = Polygon(p1, p2, p3, fill_color=PRISM_COLOR, fill_opacity=0.15, stroke_color=PRISM_COLOR, stroke_width=2)
        fronts = VGroup()
        for i in range(20):
            center = np.array([-7.5 - 0.45 * i, 0.8, 0.0])
            front = PropagatingWavefront(p1, p2, p2, p3, d_air, d_glass, d_out, center=center, extent=1.92, samples=32, stroke_width=2)
            fronts.add(front)
        self.play(FadeIn(prism), run_time=1.0)
        self.add(fronts)
        self.wait(6.5)
        note = self.change_note(note, "Im dichteren Medium läuft die Wellenfront langsamer weiter. Dadurch kippt die Front und der Lichtstrahl wird gebrochen.")
        self.wait(4.0)
        self.clean_end()

        # ══════════════════════════════════════════════════════════════════
        # INTRO 6 / ÜBERGANG ZUM TROPFEN
        # ══════════════════════════════════════════════════════════════════
        title = intro_make_title("5) Verschiedene Farben => verschiedene Brechung")
        self.play(Write(title), run_time=0.8)
        note = None
        note = self.change_note(note, "Die Materialantwort hängt von der Frequenz ab. Deshalb ist die Phasenverschiebung für Blau und Rot nicht genau gleich groß.")

        source = Dot(LEFT * 5.4 + UP * 0.8, radius=0.12, color=INTRO_WHITEISH)
        source_label = Text("weißes Licht", font=MONO, font_size=INTRO_SMALL_SIZE, color=INTRO_WHITEISH).next_to(source, UP, buff=0.16)
        drop = Circle(radius=1.0, color=INTRO_STRUCT, stroke_width=3, fill_color="#1A2533", fill_opacity=0.35).move_to(ORIGIN + RIGHT * 0.2)
        drop_label = Text("Wassertropfen", font=MONO, font_size=INTRO_SMALL_SIZE, color=INTRO_SECONDARY).next_to(drop, UP, buff=0.16)
        ray_in = Arrow(source.get_center(), drop.get_left() + UP * 0.1, buff=0, color=INTRO_WHITEISH, stroke_width=5)
        ray_red = Arrow(drop.get_right() + DOWN * 0.05, RIGHT * 5.5 + DOWN * 0.45, buff=0, color="#FF6B6B", stroke_width=5)
        ray_blue = Arrow(drop.get_right() + DOWN * 0.05, RIGHT * 5.0 + DOWN * 1.3, buff=0, color="#60A5FA", stroke_width=5)
        self.play(FadeIn(source), Write(source_label), FadeIn(drop), Write(drop_label), GrowArrow(ray_in), run_time=1.1)
        self.play(GrowArrow(ray_red), GrowArrow(ray_blue), run_time=1.2)
        note = self.change_note(note, "Blaues Licht wird im Wasser stärker gebrochen als rotes. Im Regentropfen trennt das die Farben auf.")
        self.wait(1.8)
        self.clean_end()

        # ══════════════════════════════════════════════════════════════════
        # WASSERTROPFEN / RAYTRACING
        # ══════════════════════════════════════════════════════════════════
        title = intro_make_title("Lichtweg im Tropfen: Brechung, Reflexion, Austritt")
        self.play(Write(title), run_time=0.8)
        note = None
        note = self.change_note(note, "Jetzt verfolgen wir Strahlen im Tropfen. Ein Strahl wird beim Eintritt gebrochen, innen reflektiert und beim Austritt erneut gebrochen.")

        droplet = build_droplet()
        self.add(droplet)
        DEMO_WAVELENGTH = 540.0
        wavelength = ValueTracker(DEMO_WAVELENGTH)
        y_offset = ValueTracker(0.95)
        beam = always_redraw(lambda: build_beam_group(wavelength.get_value(), y_offset.get_value()))
        beam.set_z_index(6)
        wavelength_caption = Text(f"λ = {int(DEMO_WAVELENGTH)} nm", font=MONO, font_size=26, color=WHITE).to_corner(UL, buff=0.55).set_z_index(8)
        self.play(FadeIn(beam), Write(wavelength_caption), run_time=1.2)

        angle_marker = always_redraw(lambda: build_angle_marker(wavelength.get_value(), y_offset.get_value()))
        self.add(angle_marker)

        for color_name, wavelength_nm in SELECTED_COLORS:
            angle_readout = always_redraw(
                lambda w=wavelength_nm: Text(
                    f"{(trace_ray_bundle(w, y_offset.get_value())['primary']['angle_deg']):.2f}°"
                    if trace_ray_bundle(w, y_offset.get_value())["primary"] is not None else "—",
                    font=MONO, font_size=28, color=WHITE,
                ).to_corner(UR, buff=0.55).set_z_index(8)
            )
            color_label = Text(f"{color_name}  λ = {int(wavelength_nm)} nm", font=MONO, font_size=24, color=WHITE).to_corner(UL, buff=0.55).shift(DOWN * 0.55)
            self.play(wavelength.animate.set_value(wavelength_nm), y_offset.animate.set_value(0.05), Write(angle_readout), Write(color_label), run_time=1.0)
            self.play(y_offset.animate.set_value(Y_MAX), run_time=5.5, rate_func=linear)
            self.wait(0.3)
            self.play(FadeOut(angle_readout), FadeOut(color_label), run_time=0.4)

        note = self.change_note(note, "Je nach Einfallshöhe und Wellenlänge verlassen die Strahlen den Tropfen unter leicht unterschiedlichen Winkeln. Genau daraus entsteht die Farbtrennung des Regenbogens.")
        self.play(FadeOut(beam), FadeOut(wavelength_caption), FadeOut(angle_marker), FadeOut(droplet), run_time=0.9)
        self.wait(0.2)

        # ══════════════════════════════════════════════════════════════════
        # EINZELKURVEN
        # ══════════════════════════════════════════════════════════════════
        title = intro_make_title("Ablenkwinkel für einzelne Farben")
        self.play(Write(title), run_time=0.8)
        note = None
        note = self.change_note(note, "Für jede Farbe kann man den Ablenkwinkel als Funktion der Einfallshöhe betrachten.")
        for color_name, wavelength_nm in SELECTED_COLORS:
            accent = wavelength_to_rgb(wavelength_nm)
            graph = build_final_graph(color_name, wavelength_nm, accent)
            graph.center()
            self.play(FadeIn(graph), run_time=0.9)
            self.wait(2.2)
            self.play(FadeOut(graph), run_time=0.7)
        self.clean_end()

        # ══════════════════════════════════════════════════════════════════
        # VERGLEICHSGRAPH + SCHLUSS
        # ══════════════════════════════════════════════════════════════════
        title = intro_make_title("Dispersionsvergleich und Fazit")
        self.play(Write(title), run_time=0.8)
        note = None
        note = self.change_note(note, "Im direkten Vergleich sieht man: Die Kurven der Farben liegen nicht exakt übereinander. Dadurch erscheint der Regenbogen farbig aufgespalten.")

        all_samples = []
        for color_name, wavelength_nm in SELECTED_COLORS:
            all_samples.append((color_name, wavelength_nm, compute_angle_curve(wavelength_nm)))
        all_y = [y for _, _, pts in all_samples for _, y in pts]
        y_min_global = np.floor(min(all_y) - 0.5)
        y_max_global = np.ceil(max(all_y) + 0.5)
        y_tick_cmp = round((y_max_global - y_min_global) / 4)
        if y_tick_cmp < 1:
            y_tick_cmp = 1

        cmp_axes = Axes(
            x_range=[0, Y_MAX, round(Y_MAX / 4, 2)],
            y_range=[y_min_global, y_max_global, y_tick_cmp],
            x_length=9.5,
            y_length=5.5,
            axis_config={
                "color": GREY_B,
                "stroke_opacity": 0.7,
                "stroke_width": 2.0,
                "include_numbers": False,
            },
            tips=False,
        )
        cmp_axes.center()
        cmp_x_label = Text("Einfallshöhe y [m]", font=MONO, font_size=24, color=GREY_A).next_to(cmp_axes, DOWN, buff=0.45)
        cmp_y_label = Text("Ablenkwinkel [°]", font=MONO, font_size=24, color=GREY_A)
        cmp_y_label.rotate(PI / 2)
        cmp_y_label.next_to(cmp_axes, LEFT, buff=0.55)
        cmp_title = Text("Dispersionsvergleich", font=MONO, font_size=32, color=WHITE).next_to(cmp_axes, UP, buff=0.45)

        legend_items = VGroup()
        curves = VGroup()
        for color_name, wavelength_nm, pts in all_samples:
            accent = wavelength_to_rgb(wavelength_nm)
            curve_pts = [cmp_axes.c2p(x, y) for x, y in pts]
            c = VMobject(color=accent)
            c.set_points_as_corners(curve_pts)
            c.set_stroke(color=accent, width=5.0)
            curves.add(c)
            leg = Text(f"{color_name}  {int(wavelength_nm)} nm", font=MONO, font_size=22, color=accent)
            legend_items.add(leg)
        legend_items.arrange(DOWN, aligned_edge=LEFT, buff=0.28)
        legend_items.next_to(cmp_axes, RIGHT, buff=0.55)
        cmp_group = VGroup(cmp_axes, cmp_x_label, cmp_y_label, cmp_title, curves, legend_items)

        self.play(FadeIn(cmp_axes), Write(cmp_x_label), Write(cmp_y_label), Write(cmp_title), run_time=1.0)
        for c in curves:
            self.play(Create(c), run_time=1.2, rate_func=linear)
        self.play(LaggedStart(*[Write(m) for m in legend_items], lag_ratio=0.12), run_time=0.9)
        self.wait(2.2)
        note = self.change_note(note, "Fazit: Der Regenbogen entsteht durch Brechung beim Eintritt, innere Reflexion, erneute Brechung beim Austritt und die wellenlängenabhängige Dispersion des Wassers.")
        self.wait(1.5)
        self.clean_end()
