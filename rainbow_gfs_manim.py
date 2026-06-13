from manim import *
import numpy as np

# Render command:
# manim -pql rainbow_gfs_merged.py FullRainbowAnimation
# manim -pqh rainbow_gfs_merged.py FullRainbowAnimation

config.background_color = BLACK

MONO = "DejaVu Sans Mono"
WATER_A = 1.324
WATER_B = 0.00312
AIR_REFRACTIVE_INDEX = 1.0
RADIUS = 2.2
START_X = -7.2
OUT_LENGTH = 6.4
CENTER = np.array([0.0, 0.0, 0.0])
SCENE_SHIFT = LEFT * 2.9
Y_MAX = RADIUS - 1e-3

PRIMARY = "#58C4DD"
SECONDARY = "#83C167"
ACCENT = "#FFD166"
TEXT = "#F5F7FA"
MUTED = "#B7C0CC"
SOFT = "#7F8A99"
WATER_FILL = "#123B63"
WATER_EDGE = "#4EA8DE"
SUN = "#FFD54F"
PANEL_FILL = "#0C121A"
GROUND = "#0A1118"
DARK_BAND = "#212833"

SELECTED_COLORS = [
    ("Blau", 450.0),
    ("Grün", 540.0),
    ("Rot", 650.0),
]

# ── Frame safe-area constants (Manim default: 14.22 wide × 8.0 tall) ──
FRAME_W = 14.22
FRAME_H = 8.0
SAFE_X = FRAME_W / 2 - 0.25   # ≈ 6.86
SAFE_Y = FRAME_H / 2 - 0.25   # ≈ 3.75


# ──────────────────────────────────────────────────────────────────────────────
# Physics / Raytracing
# ──────────────────────────────────────────────────────────────────────────────

def scene_point(point: np.ndarray) -> np.ndarray:
    return point + SCENE_SHIFT


def wavelength_to_rgb(wavelength_nm: float) -> ManimColor:
    w = float(np.clip(wavelength_nm, 380, 780))
    if 380 <= w < 440:
        r = -(w - 440) / (440 - 380); g = 0.0; b = 1.0
    elif 440 <= w < 490:
        r = 0.0; g = (w - 440) / (490 - 440); b = 1.0
    elif 490 <= w < 510:
        r = 0.0; g = 1.0; b = -(w - 510) / (510 - 490)
    elif 510 <= w < 580:
        r = (w - 510) / (580 - 510); g = 1.0; b = 0.0
    elif 580 <= w < 645:
        r = 1.0; g = -(w - 645) / (645 - 580); b = 0.0
    else:
        r = 1.0; g = 0.0; b = 0.0
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
    return v if norm == 0 else v / norm


def reflect(v: np.ndarray, normal: np.ndarray) -> np.ndarray:
    return normalize(v - 2 * np.dot(v, normal) * normal)


def refract(incident, normal, n1, n2):
    incident = normalize(incident); normal = normalize(normal)
    eta = n1 / n2
    cos_i = -np.dot(normal, incident)
    sin_t2 = eta * eta * (1.0 - cos_i * cos_i)
    if sin_t2 > 1.0:
        return None
    cos_t = np.sqrt(max(0.0, 1.0 - sin_t2))
    return normalize(eta * incident + (eta * cos_i - cos_t) * normal)


def ray_circle_intersection(start, direction, radius):
    direction = normalize(direction)
    p = start[:2]; d = direction[:2]
    b = 2.0 * np.dot(p, d)
    c = np.dot(p, p) - radius * radius
    disc = b * b - 4.0 * c
    if disc < 0:
        return start
    root = np.sqrt(disc)
    t1 = (-b - root) / 2.0; t2 = (-b + root) / 2.0
    candidates = [t for t in (t1, t2) if t > 1e-5]
    if not candidates:
        return start
    t = min(candidates)
    hit = p + d * t
    return np.array([hit[0], hit[1], 0.0])


def angle_between(v1, v2):
    a = normalize(v1); b = normalize(v2)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return float(np.degrees(np.arccos(dot)))


def trace_ray_bundle(wavelength_nm, y_offset, target_exit_reflections=1, max_reflections=3):
    color = wavelength_to_rgb(wavelength_nm)
    n_water = water_refractive_index(wavelength_nm)
    y = float(np.clip(y_offset, 0.0, Y_MAX))
    start = np.array([START_X, y, 0.0])
    entry = np.array([-np.sqrt(max(RADIUS * RADIUS - y * y, 0.0)), y, 0.0])
    incident = normalize(entry - start)
    entry_normal = normalize(entry - CENTER)
    incoming_segments = [(start, entry, 1.0, WHITE, 5.2)]
    inside_segments = []; reflection_segments = []; exit_segments = []
    preferred = None
    surface_reflection = reflect(incident, entry_normal)
    reflection_segments.append((entry, entry + surface_reflection * 2.0, 0.16, WHITE, 2.0))
    inside_dir = refract(incident, entry_normal, AIR_REFRACTIVE_INDEX, n_water)
    if inside_dir is None:
        return {"color": color, "incoming": incoming_segments, "inside": inside_segments,
                "reflections": reflection_segments, "exits": exit_segments, "preferred": preferred, "entry": entry}
    current_start = entry; current_dir = inside_dir; reflections_done = 0
    for _ in range(max_reflections + 3):
        hit = ray_circle_intersection(current_start + current_dir * 1e-3, current_dir, RADIUS)
        opacity = max(0.40, 1.0 - reflections_done * 0.18)
        inside_segments.append((current_start, hit, opacity, color, 4.6))
        inward_normal = -normalize(hit - CENTER)
        if reflections_done == target_exit_reflections:
            exit_dir = refract(current_dir, inward_normal, n_water, AIR_REFRACTIVE_INDEX)
            if exit_dir is not None:
                exit_segments.append((hit, hit + exit_dir * OUT_LENGTH, 1.0, color, 5.0))
                preferred = {"point": hit, "dir": exit_dir,
                             "angle_deg": angle_between(exit_dir, LEFT), "y_offset": y}
            break
        reflected_dir = reflect(current_dir, inward_normal)
        reflection_segments.append((hit, hit + reflected_dir * 0.56,
                                    max(0.22, 0.75 - 0.12 * reflections_done), color, 2.6))
        current_start = hit; current_dir = reflected_dir; reflections_done += 1
    return {"color": color, "incoming": incoming_segments, "inside": inside_segments,
            "reflections": reflection_segments, "exits": exit_segments, "preferred": preferred, "entry": entry}


def compute_angle_curve(wavelength_nm, target_exit_reflections=1, n_steps=120):
    points = []
    for i in range(n_steps + 1):
        y = Y_MAX * i / n_steps
        data = trace_ray_bundle(wavelength_nm, y, target_exit_reflections=target_exit_reflections)
        if data["preferred"] is not None:
            points.append((y, data["preferred"]["angle_deg"]))
    return points


def find_stationary_exit(wavelength_nm, target_exit_reflections=1):
    samples = compute_angle_curve(wavelength_nm, target_exit_reflections=target_exit_reflections, n_steps=140)
    samples = [s for s in samples if s[0] > 0.05]
    if not samples:
        return 0.8, 0.0
    angles = np.array([a for _, a in samples])
    idx = int(np.argmax(angles) if target_exit_reflections == 1 else np.argmin(angles))
    return samples[idx]


# ──────────────────────────────────────────────────────────────────────────────
# Drawing helpers
# ──────────────────────────────────────────────────────────────────────────────

def glow_line(start, end, color, base_width=5.0, opacity=1.0, z_index=5):
    start = scene_point(start); end = scene_point(end)
    layers = VGroup(
        Line(start, end, stroke_color=color, stroke_width=base_width * 3.0, stroke_opacity=0.06 * opacity),
        Line(start, end, stroke_color=color, stroke_width=base_width * 1.6, stroke_opacity=0.18 * opacity),
        Line(start, end, stroke_color=color, stroke_width=base_width, stroke_opacity=0.98 * opacity),
    )
    layers.set_z_index(z_index)
    return layers


def build_beam_group(wavelength_nm, y_offset, target_exit_reflections=1):
    bundle = trace_ray_bundle(wavelength_nm, y_offset, target_exit_reflections=target_exit_reflections)
    group = VGroup()
    for seg_group in (bundle["incoming"], bundle["inside"], bundle["exits"], bundle["reflections"]):
        for start, end, opacity, color, width in seg_group:
            group.add(glow_line(start, end, color, base_width=width, opacity=opacity,
                                z_index=6 if width >= 4.0 else 5))
    return group


def build_droplet():
    core = Circle(radius=RADIUS, color=WATER_EDGE).move_to(SCENE_SHIFT)
    core.set_stroke(width=2.0, opacity=0.82)
    core.set_fill(color=WATER_FILL, opacity=0.12)
    glow_1 = Circle(radius=RADIUS * 1.01, color=BLUE_B).move_to(SCENE_SHIFT)
    glow_1.set_stroke(width=15, opacity=0.08)
    glow_2 = Circle(radius=RADIUS * 1.03, color=BLUE_A).move_to(SCENE_SHIFT)
    glow_2.set_stroke(width=32, opacity=0.03)
    inner = Circle(radius=RADIUS * 0.94, color=WHITE).move_to(SCENE_SHIFT)
    inner.set_stroke(width=1.2, opacity=0.14)
    highlight = Arc(radius=RADIUS * 0.96, start_angle=PI * 0.10, angle=PI * 0.46, color=WHITE)
    highlight.set_stroke(width=3, opacity=0.18)
    highlight.move_arc_center_to(SCENE_SHIFT + LEFT * 0.08 + UP * 0.10)
    droplet = VGroup(glow_2, glow_1, core, inner, highlight)
    droplet.set_z_index(2)
    return droplet


def rainbow_arc(radius=2.6, center=ORIGIN, reverse=False, stroke_width=12,
                start_angle=20 * DEGREES, angle=140 * DEGREES):
    colors = [wavelength_to_rgb(w) for w in [650, 600, 570, 540, 490, 450]]
    if reverse:
        colors = list(reversed(colors))
    arcs = VGroup()
    for i, color in enumerate(colors):
        arc = Arc(radius=radius - i * 0.11, start_angle=start_angle, angle=angle,
                  color=color, stroke_width=stroke_width)
        arc.move_arc_center_to(center)
        arcs.add(arc)
    return arcs


def build_eye(scale=1.0, color=TEXT):
    outer = Ellipse(width=1.6 * scale, height=0.82 * scale, color=color, stroke_width=3)
    pupil = Dot(radius=0.10 * scale, color=PRIMARY)
    return VGroup(outer, pupil)


def fit_text(text_obj, max_width):
    if text_obj.width > max_width:
        text_obj.scale_to_fit_width(max_width)
    return text_obj


def formula_card(tex_string, color=TEXT, font_size=36):
    formula = MathTex(tex_string, font_size=font_size, color=color)
    box = RoundedRectangle(corner_radius=0.16, width=formula.width + 0.55,
                           height=formula.height + 0.40, color=color)
    box.set_fill(PANEL_FILL, opacity=0.95)
    box.set_stroke(color, width=2.0, opacity=0.75)
    box.move_to(formula)
    return VGroup(box, formula)


def angle_label(tex, color, position, font_size=22):
    label = MathTex(tex, color=color, font_size=font_size)
    back = BackgroundRectangle(label, color=BLACK, fill_opacity=0.7, buff=0.08)
    group = VGroup(back, label)
    group.move_to(position)
    return group


def section_title(text, font_size=28, color=ACCENT):
    """Small title card shown briefly at start of each section."""
    label = Text(text, font=MONO, font_size=font_size, color=color)
    box = RoundedRectangle(corner_radius=0.14, width=label.width + 0.6,
                           height=label.height + 0.32, color=color)
    box.set_fill(PANEL_FILL, opacity=0.92)
    box.set_stroke(color, width=1.6, opacity=0.80)
    box.move_to(label)
    grp = VGroup(box, label).to_edge(UP, buff=0.22)
    return grp


def clean_transition(scene, keep=None, run_time=0.55):
    """Fade out everything except objects in `keep`."""
    to_remove = [m for m in scene.mobjects if keep is None or m not in keep]
    if to_remove:
        scene.play(FadeOut(Group(*to_remove)), run_time=run_time)
    scene.wait(0.15)


# ──────────────────────────────────────────────────────────────────────────────
# Merged scene
# ──────────────────────────────────────────────────────────────────────────────

class FullRainbowAnimation(Scene):
    def construct(self):
        self.camera.background_color = BLACK

        self._scene_angle_hook()
        self._scene_refraction_and_reflection()
        self._scene_single_drop_physics()
        self._scene_angle_selection_graph()
        self._scene_why_arc_not_spot()
        self._scene_primary_secondary_rainbow()
        self._scene_final_takeaway()

    # ── helpers ──────────────────────────────────────────────────────────────

    def _fade_all_out(self, run_time=0.55):
        if self.mobjects:
            self.play(FadeOut(Group(*self.mobjects)), run_time=run_time)
        self.wait(0.18)

    def _show_title(self, text):
        title = section_title(text)
        self.play(FadeIn(title, shift=DOWN * 0.08), run_time=0.40)
        self.wait(0.30)
        self.play(FadeOut(title), run_time=0.30)

    # ── Scene 1: AngleHook ───────────────────────────────────────────────────

    def _scene_angle_hook(self):
        self._show_title("1 · Der Regenbogenwinkel")

        # observer and sun — keep within frame
        # Original: observer at RIGHT*4.85 + DOWN*1.50, sun at RIGHT*5.45 + UP*2.55
        # RIGHT*5.45 is fine (< 6.86). UP*2.55 fine (< 3.75). Keep as-is.
        observer = build_eye(1.05).move_to(RIGHT * 4.85 + DOWN * 1.50)
        sun = (Circle(radius=0.32, color=SUN, stroke_width=0)
               .set_fill(SUN, opacity=1)
               .move_to(RIGHT * 5.10 + UP * 2.40))  # shifted slightly inward
        sun_rays = VGroup(*[
            Line(sun.get_center() + rotate_vector(RIGHT * 0.44, ang),
                 sun.get_center() + rotate_vector(RIGHT * 0.72, ang),
                 color=SUN, stroke_width=2.1)
            for ang in np.linspace(0, TAU, 10, endpoint=False)
        ])

        axis = DashedLine(observer.get_center(), observer.get_center() + LEFT * 5.5,
                          dash_length=0.11, color=SOFT, stroke_width=1.8)
        antisolar = Dot(observer.get_center() + LEFT * 4.75, radius=0.05, color=WHITE)

        grey_dirs = [rotate_vector(LEFT, deg * DEGREES) for deg in [36, 38, 40, 42, 44, 46]]
        candidate_rays = VGroup(*[
            Line(observer.get_center(), observer.get_center() + d * 5.0,
                 color=GREY_C, stroke_width=2.1, stroke_opacity=0.34)
            for d in grey_dirs
        ])

        blue_dir = rotate_vector(LEFT, 40 * DEGREES)
        red_dir = rotate_vector(LEFT, 42 * DEGREES)
        blue_ray = Line(observer.get_center(), observer.get_center() + blue_dir * 5.0,
                        color=wavelength_to_rgb(450), stroke_width=5.0)
        red_ray = Line(observer.get_center(), observer.get_center() + red_dir * 5.15,
                       color=wavelength_to_rgb(650), stroke_width=5.0)

        blue_drops = VGroup(*[
            Circle(radius=0.10, color=wavelength_to_rgb(450), stroke_width=1.3)
            .set_fill(wavelength_to_rgb(450), opacity=0.16)
            .move_to(observer.get_center() + blue_dir * d)
            for d in [2.0, 2.8, 3.6, 4.4]
        ])
        red_drops = VGroup(*[
            Circle(radius=0.10, color=wavelength_to_rgb(650), stroke_width=1.3)
            .set_fill(wavelength_to_rgb(650), opacity=0.16)
            .move_to(observer.get_center() + red_dir * d)
            for d in [2.2, 3.0, 3.8, 4.6]
        ])

        left_helper = Line(observer.get_center(), observer.get_center() + LEFT * 1.2, stroke_opacity=0)
        blue_helper = Line(observer.get_center(), observer.get_center() + blue_dir * 1.0, stroke_opacity=0)
        red_helper = Line(observer.get_center(), observer.get_center() + red_dir * 1.0, stroke_opacity=0)
        blue_angle = Angle(left_helper, blue_helper, radius=0.46,
                           color=wavelength_to_rgb(450), stroke_width=3.0)
        red_angle = Angle(left_helper, red_helper, radius=0.74,
                          color=wavelength_to_rgb(650), stroke_width=3.0)
        blue_label = angle_label(r"40^\circ", wavelength_to_rgb(450),
                                 observer.get_center() + LEFT * 0.84 + UP * 0.52)
        red_label = angle_label(r"42^\circ", wavelength_to_rgb(650),
                                observer.get_center() + LEFT * 0.90 + UP * 0.92)

        self.play(FadeIn(sun), FadeIn(sun_rays), FadeIn(observer), run_time=0.8)
        self.play(Create(axis), FadeIn(antisolar), run_time=0.7)
        self.wait(0.2)
        self.play(LaggedStart(*[Create(r) for r in candidate_rays], lag_ratio=0.12), run_time=1.1)
        self.play(FadeIn(blue_ray), FadeIn(red_ray),
                  candidate_rays.animate.set_opacity(0.12), run_time=0.9)
        self.play(FadeIn(blue_drops, lag_ratio=0.08), FadeIn(red_drops, lag_ratio=0.08), run_time=0.8)
        self.play(Create(blue_angle), Create(red_angle),
                  FadeIn(blue_label), FadeIn(red_label), run_time=0.8)
        self.wait(1.7)
        self._fade_all_out()

    # ── Scene 2: RefractionAndReflection ─────────────────────────────────────

    def _scene_refraction_and_reflection(self):
        self._show_title("2 · Brechung & Reflexion")

        top_region = (Rectangle(width=6.8, height=2.45, stroke_width=0)
                      .set_fill("#10161F", opacity=0.92)
                      .move_to(RIGHT * 1.9 + UP * 1.5))
        bottom_region = (Rectangle(width=6.8, height=3.1, stroke_width=0)
                         .set_fill("#12304A", opacity=0.42)
                         .move_to(RIGHT * 1.9 + DOWN * 1.35))
        interface = Line(LEFT * 0.8 + DOWN * 0.1, RIGHT * 4.6 + DOWN * 0.1,
                         color=GREY_B, stroke_width=2.6)

        hit = np.array([1.9, -0.1, 0.0])
        normal = DashedLine(hit + DOWN * 2.5, hit + UP * 2.6,
                            dash_length=0.12, color=SOFT, stroke_width=1.8)

        inc_dir = np.array([np.cos(140 * DEGREES), np.sin(140 * DEGREES), 0.0])
        refl_dir = np.array([np.cos(40 * DEGREES), np.sin(40 * DEGREES), 0.0])
        refr_dir = np.array([np.cos(-55 * DEGREES), np.sin(-55 * DEGREES), 0.0])

        inc_start = hit + inc_dir * 2.8
        refl_end = hit + refl_dir * 2.8
        refr_end = hit + refr_dir * 2.6

        incoming = Arrow(inc_start, hit, buff=0, color=YELLOW_B, stroke_width=3.6,
                         max_tip_length_to_length_ratio=0.07)
        reflected = Arrow(hit, refl_end, buff=0, color=ACCENT, stroke_width=3.6,
                          max_tip_length_to_length_ratio=0.07)
        refracted = Arrow(hit, refr_end, buff=0, color=PRIMARY, stroke_width=3.6,
                          max_tip_length_to_length_ratio=0.07)

        inc_arc = Arc(radius=0.55, start_angle=90 * DEGREES, angle=50 * DEGREES,
                      color=YELLOW_B, stroke_width=2.8).move_arc_center_to(hit)
        refl_arc = Arc(radius=0.74, start_angle=40 * DEGREES, angle=50 * DEGREES,
                       color=ACCENT, stroke_width=2.8).move_arc_center_to(hit)
        refr_arc = Arc(radius=0.58, start_angle=-90 * DEGREES, angle=35 * DEGREES,
                       color=PRIMARY, stroke_width=2.8).move_arc_center_to(hit)

        theta_i = MathTex(r"\theta_i", color=YELLOW_B, font_size=24).move_to(hit + UP * 0.88 + LEFT * 0.40)
        theta_r = MathTex(r"\theta_r", color=ACCENT, font_size=24).move_to(hit + UP * 0.88 + RIGHT * 0.44)
        theta_t = MathTex(r"\theta_t", color=PRIMARY, font_size=24).move_to(hit + DOWN * 0.88 + RIGHT * 0.44)

        formula_reflection = formula_card(r"\theta_i = \theta_r", color=ACCENT, font_size=34)
        formula_snell = formula_card(r"n_1\sin\theta_1 = n_2\sin\theta_2", color=PRIMARY, font_size=30)
        formulas = VGroup(formula_reflection, formula_snell).arrange(DOWN, buff=0.24, aligned_edge=LEFT)
        formulas.to_edge(RIGHT, buff=0.5).shift(DOWN * 1.35)
        # Safety: keep inside frame
        if formulas.get_right()[0] > SAFE_X:
            formulas.shift(LEFT * (formulas.get_right()[0] - SAFE_X))

        self.play(FadeIn(top_region), FadeIn(bottom_region), run_time=0.7)
        self.play(Create(interface), Create(normal), run_time=0.7)
        self.play(GrowArrow(incoming), run_time=0.7)
        self.play(GrowArrow(reflected), Create(inc_arc), Create(refl_arc),
                  FadeIn(theta_i), FadeIn(theta_r), run_time=0.9)
        self.play(FadeIn(formula_reflection), run_time=0.6)
        self.wait(0.4)
        self.play(GrowArrow(refracted), Create(refr_arc), FadeIn(theta_t), run_time=0.8)
        self.play(FadeIn(formula_snell), run_time=0.6)
        self.wait(1.8)
        self._fade_all_out()

    # ── Scene 3: SingleDropPhysics ────────────────────────────────────────────

    def _scene_single_drop_physics(self):
        self._show_title("3 · Ein Tropfen, drei Farben")

        droplet = build_droplet()
        red_y, red_angle = find_stationary_exit(650.0, 1)
        green_y, green_angle = find_stationary_exit(540.0, 1)
        blue_y, blue_angle = find_stationary_exit(450.0, 1)

        y_tracker = ValueTracker(blue_y - 0.18)
        wl_tracker = ValueTracker(450.0)
        animated_beam = always_redraw(
            lambda: build_beam_group(wl_tracker.get_value(), y_tracker.get_value(),
                                     target_exit_reflections=1))

        red_group = build_beam_group(650.0, red_y, target_exit_reflections=1)
        green_group = build_beam_group(540.0, green_y, target_exit_reflections=1)
        blue_group = build_beam_group(450.0, blue_y, target_exit_reflections=1)

        red_data = trace_ray_bundle(650.0, red_y, target_exit_reflections=1)["preferred"]
        green_data = trace_ray_bundle(540.0, green_y, target_exit_reflections=1)["preferred"]
        blue_data = trace_ray_bundle(450.0, blue_y, target_exit_reflections=1)["preferred"]

        red_label = angle_label(
            rf"{red_angle:.1f}^\circ", wavelength_to_rgb(650),
            scene_point(red_data["point"] + normalize(red_data["dir"]) * 2.45 + np.array([0.08, 0.52, 0.0])),
            font_size=23)
        green_label = angle_label(
            rf"{green_angle:.1f}^\circ", wavelength_to_rgb(540),
            scene_point(green_data["point"] + normalize(green_data["dir"]) * 2.15 + np.array([0.18, 0.05, 0.0])),
            font_size=23)
        blue_label = angle_label(
            rf"{blue_angle:.1f}^\circ", wavelength_to_rgb(450),
            scene_point(blue_data["point"] + normalize(blue_data["dir"]) * 2.05 + np.array([0.02, -0.48, 0.0])),
            font_size=23)

        self.play(FadeIn(droplet), run_time=0.8)
        self.play(FadeIn(animated_beam), run_time=0.6)
        self.wait(0.25)
        self.play(y_tracker.animate.set_value(red_y + 0.14),
                  wl_tracker.animate.set_value(650.0),
                  run_time=3.8, rate_func=there_and_back)
        self.play(y_tracker.animate.set_value(green_y),
                  wl_tracker.animate.set_value(540.0), run_time=1.3)
        self.wait(0.2)
        self.play(FadeOut(animated_beam),
                  FadeIn(blue_group), FadeIn(green_group), FadeIn(red_group), run_time=1.0)
        self.play(FadeIn(blue_label), FadeIn(green_label), FadeIn(red_label), run_time=0.6)
        self.wait(1.8)
        self._fade_all_out()

    # ── Scene 4: AngleSelectionGraph ─────────────────────────────────────────

    def _scene_angle_selection_graph(self):
        self._show_title("4 · Winkelkurven")

        all_samples = [(name, wl, compute_angle_curve(wl, target_exit_reflections=1, n_steps=140))
                       for name, wl in SELECTED_COLORS]
        angle_values = [angle for _, _, pts in all_samples for _, angle in pts]
        y_min = np.floor(min(angle_values) - 0.35)
        y_max_v = np.ceil(max(angle_values) + 0.35)

        axes = Axes(
            x_range=[0, Y_MAX, round(Y_MAX / 4, 2)],
            y_range=[y_min, y_max_v, 1],
            x_length=7.4,
            y_length=4.8,
            axis_config={"color": GREY_B, "stroke_opacity": 0.72, "stroke_width": 2.0,
                         "include_numbers": False},
            tips=False,
        ).move_to(LEFT * 2.0 + DOWN * 0.2)

        x_tick_marks = VGroup(*[
            Line(axes.c2p(val, y_min) + UP * 0.08,
                 axes.c2p(val, y_min) + DOWN * 0.08, color=SOFT, stroke_width=1.6)
            for val in [0.0, round(Y_MAX / 2, 2), round(Y_MAX, 2)]
        ])
        y_tick_marks = VGroup(*[
            Line(axes.c2p(0, val) + LEFT * 0.08,
                 axes.c2p(0, val) + RIGHT * 0.08, color=SOFT, stroke_width=1.6)
            for val in [40, 41, 42] if y_min <= val <= y_max_v
        ])
        x_tick_labels = VGroup(*[
            MathTex(label, color=SOFT, font_size=20).next_to(axes.c2p(val, y_min), DOWN, buff=0.16)
            for val, label in [(0.0, "0"), (round(Y_MAX / 2, 2), r"R/2"), (round(Y_MAX, 2), r"R")]
        ])
        y_tick_labels = VGroup(*[
            MathTex(rf"{int(val)}^\circ", color=SOFT, font_size=20)
            .next_to(axes.c2p(0, val), LEFT, buff=0.18)
            for val in [40, 41, 42] if y_min <= val <= y_max_v
        ])

        curves = VGroup()
        stationary_items = VGroup()
        label_offsets = {
            "Rot": np.array([0.72, 0.34, 0.0]),
            "Grün": np.array([0.72, -0.02, 0.0]),
            "Blau": np.array([0.72, -0.38, 0.0]),
        }
        for name, wl, pts in all_samples:
            color = wavelength_to_rgb(wl)
            curve_points = [axes.c2p(x, y) for x, y in pts]
            glow = VMobject(color=color)
            glow.set_points_as_corners(curve_points)
            glow.set_stroke(color=color, width=12, opacity=0.10)
            curve = VMobject(color=color)
            curve.set_points_as_corners(curve_points)
            curve.set_stroke(color=color, width=4.8)
            curves.add(glow, curve)
            stat_y, stat_angle = find_stationary_exit(wl, 1)
            dot = Dot(axes.c2p(stat_y, stat_angle), radius=0.055, color=color)
            lbl = angle_label(rf"{stat_angle:.1f}^\circ", color,
                              axes.c2p(stat_y, stat_angle) + label_offsets[name], font_size=22)
            stationary_items.add(dot, lbl)

        # Clamp right-side labels inside frame
        for item in stationary_items:
            if item.get_right()[0] > SAFE_X:
                item.shift(LEFT * (item.get_right()[0] - SAFE_X))

        self.play(FadeIn(axes), FadeIn(x_tick_marks), FadeIn(y_tick_marks),
                  FadeIn(x_tick_labels), FadeIn(y_tick_labels), run_time=0.8)
        for i in range(0, len(curves), 2):
            self.play(FadeIn(curves[i]), Create(curves[i + 1]), run_time=0.95)
        self.wait(0.35)
        for i in range(0, len(stationary_items), 2):
            self.play(FadeIn(stationary_items[i]), FadeIn(stationary_items[i + 1]), run_time=0.45)
        self.wait(1.9)
        self._fade_all_out()

    # ── Scene 5: WhyArcNotSpot ────────────────────────────────────────────────

    def _scene_why_arc_not_spot(self):
        self._show_title("5 · Warum ein Bogen?")

        center = DOWN * 2.02
        radius = 3.62
        # observer and sun stay within frame: observer at DOWN*1.42, radius 3.62 means
        # center + UP*3.62 = UP*1.60 which is fine.
        observer = build_eye(0.95).move_to(DOWN * 1.42)
        sun = (Circle(radius=0.24, color=SUN, stroke_width=0)
               .set_fill(SUN, opacity=1)
               .move_to(RIGHT * 4.85 + DOWN * 1.62))
        axis = DashedLine(observer.get_center(), center, dash_length=0.10,
                          color=SOFT, stroke_width=1.8)
        antisolar = Dot(point=center, radius=0.055, color=WHITE)

        guide_circle = (Circle(radius=radius, color=SOFT, stroke_width=2.0, stroke_opacity=0.24)
                        .move_to(center))
        visible_arc = rainbow_arc(radius=radius, center=center, reverse=False, stroke_width=9.5,
                                   start_angle=16 * DEGREES, angle=148 * DEGREES)

        endpoint_angles = np.linspace(24 * DEGREES, 156 * DEGREES, 9)
        spokes = VGroup(*[
            DashedLine(center, center + rotate_vector(RIGHT * radius, ang),
                       dash_length=0.09, color=GREY_C, stroke_width=1.25, stroke_opacity=0.52)
            for ang in endpoint_angles
        ])
        sight_lines = VGroup(*[
            DashedLine(observer.get_center(), center + rotate_vector(RIGHT * radius, ang),
                       dash_length=0.09, color=WHITE, stroke_width=1.0, stroke_opacity=0.38)
            for ang in endpoint_angles
        ])
        rain_drops = VGroup(*[
            Dot(center + rotate_vector(RIGHT * radius, ang), radius=0.045, color=WHITE)
            for ang in endpoint_angles
        ])
        # Horizon at DOWN*2.76 — bottom of frame at DOWN*4 so fine
        horizon = Line(LEFT * 5.9 + DOWN * 2.76, RIGHT * 5.9 + DOWN * 2.76,
                       color=GREY_B, stroke_width=2.0, stroke_opacity=0.75)
        # Clamp horizon endpoints inside frame width
        if horizon.get_left()[0] < -SAFE_X:
            horizon.set_points_as_corners([
                np.array([-SAFE_X, -2.76, 0]), np.array([SAFE_X, -2.76, 0])
            ])

        self.play(FadeIn(observer), FadeIn(sun), FadeIn(antisolar), run_time=0.7)
        self.play(Create(axis), Create(guide_circle), run_time=0.85)
        self.play(Create(spokes), FadeIn(rain_drops, lag_ratio=0.05), run_time=0.95)
        self.play(Create(sight_lines), run_time=0.95)
        self.wait(0.35)
        self.play(Create(horizon), FadeIn(visible_arc), run_time=0.9)
        self.wait(1.9)
        self._fade_all_out()

    # ── Scene 6: PrimarySecondaryRainbow ─────────────────────────────────────

    def _scene_primary_secondary_rainbow(self):
        self._show_title("6 · Primär- & Sekundärbogen")

        center = DOWN * 2.02
        observer = build_eye(0.95).move_to(DOWN * 1.42)
        antisolar = Dot(point=center, radius=0.055, color=WHITE)
        axis = DashedLine(observer.get_center(), center, dash_length=0.10,
                          color=SOFT, stroke_width=1.8)

        outer_radius = 3.85
        inner_radius = 3.18
        outer_guide = (Circle(radius=outer_radius, color=SOFT, stroke_width=2.0, stroke_opacity=0.16)
                       .move_to(center))
        inner_guide = (Circle(radius=inner_radius, color=SOFT, stroke_width=2.0, stroke_opacity=0.16)
                       .move_to(center))

        primary_hint = (Circle(radius=inner_radius, color=wavelength_to_rgb(620),
                               stroke_width=3.0, stroke_opacity=0.20).move_to(center))
        secondary_hint = (Circle(radius=outer_radius, color=wavelength_to_rgb(470),
                                 stroke_width=3.0, stroke_opacity=0.18).move_to(center))
        primary_visible = rainbow_arc(radius=inner_radius, center=center, reverse=False,
                                       stroke_width=9.5, start_angle=16 * DEGREES, angle=148 * DEGREES)
        secondary_visible = rainbow_arc(radius=outer_radius, center=center, reverse=True,
                                         stroke_width=8.5, start_angle=16 * DEGREES, angle=148 * DEGREES)

        alexander_band = Arc(radius=3.50, start_angle=16 * DEGREES, angle=148 * DEGREES,
                             color=DARK_BAND, stroke_width=14)
        alexander_band.move_arc_center_to(center)
        alexander_band.set_opacity(0.42)

        primary_sight = DashedLine(
            observer.get_center(),
            center + rotate_vector(RIGHT * inner_radius, 90 * DEGREES),
            dash_length=0.09, color=wavelength_to_rgb(650), stroke_width=1.4, stroke_opacity=0.65)
        secondary_sight = DashedLine(
            observer.get_center(),
            center + rotate_vector(RIGHT * outer_radius, 90 * DEGREES),
            dash_length=0.09, color=wavelength_to_rgb(450), stroke_width=1.4, stroke_opacity=0.65)
        horizon = Line(LEFT * 5.9 + DOWN * 2.76, RIGHT * 5.9 + DOWN * 2.76,
                       color=GREY_B, stroke_width=2.0, stroke_opacity=0.75)

        self.play(FadeIn(observer), FadeIn(antisolar), run_time=0.6)
        self.play(Create(axis), FadeIn(inner_guide), FadeIn(outer_guide), run_time=0.8)
        self.play(FadeIn(primary_hint), FadeIn(secondary_hint), run_time=0.7)
        self.play(Create(primary_sight), Create(secondary_sight), run_time=0.75)
        self.play(Create(horizon), FadeIn(alexander_band),
                  FadeIn(primary_visible), FadeIn(secondary_visible), run_time=1.0)
        self.wait(2.0)
        self._fade_all_out()

    # ── Scene 7: FinalTakeaway ────────────────────────────────────────────────

    def _scene_final_takeaway(self):
        self._show_title("7 · Fazit")

        line1 = Text("2× Brechung", font=MONO, font_size=34, color=TEXT)
        line2 = Text("1× innere Reflexion", font=MONO, font_size=34, color=TEXT)
        line3 = MathTex(r"dD/dy \approx 0", font_size=38, color=PRIMARY)
        line4 = Text("Regenbogen = Winkelphänomen", font=MONO, font_size=34, color=ACCENT)
        stack = VGroup(line1, line2, line3, line4).arrange(DOWN, buff=0.34).move_to(ORIGIN)

        # Safety: scale down if too wide
        if stack.width > FRAME_W - 1.0:
            stack.scale_to_fit_width(FRAME_W - 1.0)

        for item in stack:
            self.play(FadeIn(item, shift=UP * 0.08), run_time=0.42)
            self.wait(0.18)
        self.wait(2.0)
        self._fade_all_out()