from manim import *
import numpy as np

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
            "include_numbers": True,
            "font_size": 22,
            "decimal_number_config": {"num_decimal_places": 1},
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

class WaterDropletDispersion(Scene):
    def construct(self):
        # ══════════════════════════════════════════════════════════════════
        # SZENE 0: Prism Refraction (Wavefronts)
        # ══════════════════════════════════════════════════════════════════
        
        # Gigantic equilateral prism (Side 40.0)
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
        ri_lbl = Text("Einfallsstrahl", font=MONO, font_size=18, color=YELLOW_B)
        ri_lbl.next_to(ri_start, UP, buff=0.1)

        self.play(GrowArrow(ri_ray), FadeIn(ri_lbl), run_time=0.9)
        self.wait(0.2)

        ri_arc = Arc(radius=0.75, start_angle=PI / 2, angle=-(PI / 2 - ri_rad),
                     color=YELLOW_B, stroke_width=2.4).move_arc_center_to(r_hit)
        ri_angle_lbl = MathTex(r"\theta_{\mathrm{ein}}", font_size=28, color=YELLOW_B)
        ri_angle_lbl.move_to(r_hit + UP * 1.0 + LEFT * 0.55)
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

        lbl_n_comp = Text("Normal-\nkomponente", font=MONO, font_size=16, color=RED_B,
                          line_spacing=1.1)
        lbl_n_comp.next_to(comp_origin + d_n * 1.6, RIGHT, buff=0.12)

        lbl_t_comp = Text("Tangential-\nkomponente", font=MONO, font_size=16, color=GREEN_B,
                          line_spacing=1.1)
        lbl_t_comp.next_to(comp_origin + d_t * 1.6, DOWN, buff=0.12)

        decomp_title = Text("Zerlegung des Einfallsstrahls:", font=MONO, font_size=19, color=GREY_A)
        decomp_title.to_corner(UR, buff=0.55).shift(DOWN * 0.4)

        self.play(FadeIn(decomp_title), run_time=0.5)
        self.play(GrowArrow(arr_tang_comp), FadeIn(lbl_t_comp), run_time=0.7)
        self.play(GrowArrow(arr_normal_comp), FadeIn(lbl_n_comp), run_time=0.7)
        self.wait(0.4)

        rule_lbl = Text(
            "Bei Reflexion:\n"
            "  Normalkomp.   → umgekehrt  (−)\n"
            "  Tangentialkomp. → erhalten  (+)",
            font=MONO, font_size=18, color=GREY_A, line_spacing=1.4,
        )
        rule_lbl.to_corner(UR, buff=0.55).shift(DOWN * 1.6)
        self.play(FadeIn(rule_lbl), run_time=0.7)
        self.wait(0.5)

        d_refl = d_t - d_n
        ro_end = r_hit + d_refl * 2.8

        ro_ray = Arrow(r_hit, ro_end, buff=0, color=ORANGE,
                       stroke_width=3.5, max_tip_length_to_length_ratio=0.07)
        ro_lbl = Text("Reflektierter Strahl", font=MONO, font_size=18, color=ORANGE)
        ro_lbl.next_to(ro_end, UP, buff=0.1)

        ro_arc = Arc(radius=0.75, start_angle=PI / 2, angle=(PI / 2 - ri_rad),
                     color=ORANGE, stroke_width=2.4).move_arc_center_to(r_hit)
        ro_angle_lbl = MathTex(r"\theta_{\mathrm{aus}}", font_size=28, color=ORANGE)
        ro_angle_lbl.move_to(r_hit + UP * 1.0 + RIGHT * 0.6)

        self.play(GrowArrow(ro_ray), FadeIn(ro_lbl), run_time=0.9)
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
            ri_ray, ri_lbl, ri_arc, ri_angle_lbl,
            arr_inc_full, arr_normal_comp, arr_tang_comp,
            lbl_n_comp, lbl_t_comp, decomp_title, rule_lbl,
            ro_ray, ro_lbl, ro_arc, ro_angle_lbl,
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

        lbl_m1 = Text("Luft   n₁ = 1.00", font=MONO, font_size=19, color=BLUE_B)
        lbl_m1.move_to(UP * 1.7 + LEFT * 3.5)
        lbl_m2 = Text("Wasser  n₂ = 1.33", font=MONO, font_size=19, color=TEAL_B)
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
        s_inc_lbl = Text("Einfallsstrahl", font=MONO, font_size=18, color=YELLOW_B)
        s_inc_lbl.next_to(s_inc_start, UP + LEFT * 0.2, buff=0.1)

        s1_arc = Arc(radius=0.72, start_angle=PI / 2, angle=-s_inc_rad,
                     color=YELLOW_B, stroke_width=2.4).move_arc_center_to(s_hit)
        s_theta1_lbl = MathTex(r"\theta_1", font_size=28, color=YELLOW_B)
        s_theta1_lbl.move_to(s_hit + UP * 0.95 + RIGHT * 0.52)

        self.play(GrowArrow(s_inc_ray), FadeIn(s_inc_lbl), run_time=0.8)
        self.play(Create(s1_arc), Write(s_theta1_lbl), run_time=0.6)
        self.wait(0.2)

        n1, n2 = 1.0, 1.33
        sin_t2 = n1 * np.sin(s_inc_rad) / n2
        s_refr_rad = np.arcsin(sin_t2)
        s_d_refr = np.array([np.sin(s_refr_rad), -np.cos(s_refr_rad), 0.0])
        s_refr_end = s_hit + s_d_refr * 2.5

        s_refr_ray = Arrow(s_hit, s_refr_end, buff=0, color=TEAL_B,
                           stroke_width=3.5, max_tip_length_to_length_ratio=0.07)
        s_refr_lbl = Text("Gebrochener Strahl", font=MONO, font_size=18, color=TEAL_B)
        s_refr_lbl.next_to(s_refr_end, DOWN + RIGHT * 0.1, buff=0.1)

        s2_arc = Arc(radius=0.72, start_angle=-PI / 2, angle=s_refr_rad,
                     color=TEAL_B, stroke_width=2.4).move_arc_center_to(s_hit)
        s_theta2_lbl = MathTex(r"\theta_2", font_size=28, color=TEAL_B)
        s_theta2_lbl.move_to(s_hit + DOWN * 0.9 + RIGHT * 0.46)

        self.play(GrowArrow(s_refr_ray), FadeIn(s_refr_lbl), run_time=0.8)
        self.play(Create(s2_arc), Write(s_theta2_lbl), run_time=0.6)
        self.wait(0.3)

        snell_formula = MathTex(r"n_1 \sin\theta_1 = n_2 \sin\theta_2",
                                font_size=46, color=WHITE)
        snell_formula.to_corner(UR, buff=0.6).shift(DOWN * 0.5)
        self.play(Write(snell_formula), run_time=1.1)
        self.wait(0.4)

        snell_note = Text(
            f"n₂ > n₁  →  θ₂ < θ₁\n"
            f"Beispiel: θ₁ = {s_inc_deg:.0f}°  →  θ₂ = {np.degrees(s_refr_rad):.1f}°",
            font=MONO, font_size=19, color=GREY_A, line_spacing=1.4,
        )
        snell_note.to_corner(UR, buff=0.6).shift(DOWN * 2.2)
        self.play(FadeIn(snell_note), run_time=0.7)
        self.wait(2.5)

        snell_scene = VGroup(
            snell_title, s_interface, lbl_m1, lbl_m2,
            s_normal, s_normal_lbl,
            s_inc_ray, s_inc_lbl, s1_arc, s_theta1_lbl,
            s_refr_ray, s_refr_lbl, s2_arc, s_theta2_lbl,
            snell_formula, snell_note,
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
                "include_numbers": True,
                "font_size": 22,
                "decimal_number_config": {"num_decimal_places": 0},
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
