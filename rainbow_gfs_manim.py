from manim import *
import numpy as np

# Render-Beispiele:
# manim -pql rainbow_gfs_manim.py AngleHook RefractionAndReflection SingleDropPhysics AngleSelectionGraph WhyArcNotSpot PrimarySecondaryRainbow FinalTakeaway
# manim -pqh rainbow_gfs_manim.py AngleHook

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
MUTED = GREY_B
SOFT = GREY_C
WATER_FILL = "#123B63"
WATER_EDGE = "#4EA8DE"
SUN = "#FFD54F"

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


def trace_ray_bundle(
    wavelength_nm: float,
    y_offset: float,
    target_exit_reflections: int = 1,
    max_reflections: int = 3,
):
    color = wavelength_to_rgb(wavelength_nm)
    n_water = water_refractive_index(wavelength_nm)

    y = float(np.clip(y_offset, 0.0, Y_MAX))
    start = np.array([START_X, y, 0.0])
    entry = np.array([-np.sqrt(max(RADIUS * RADIUS - y * y, 0.0)), y, 0.0])
    incident = normalize(entry - start)
    entry_normal = normalize(entry - CENTER)

    incoming_segments = [(start, entry, 1.0, WHITE, 5.4)]
    inside_segments = []
    reflection_segments = []
    exit_segments = []
    preferred = None

    surface_reflection = reflect(incident, entry_normal)
    reflection_segments.append((entry, entry + surface_reflection * 2.0, 0.16, WHITE, 2.0))

    inside_dir = refract(incident, entry_normal, AIR_REFRACTIVE_INDEX, n_water)
    if inside_dir is None:
        return {
            "color": color,
            "incoming": incoming_segments,
            "inside": inside_segments,
            "reflections": reflection_segments,
            "exits": exit_segments,
            "preferred": preferred,
            "entry": entry,
        }

    current_start = entry
    current_dir = inside_dir
    reflections_done = 0

    for _ in range(max_reflections + 3):
        hit = ray_circle_intersection(current_start + current_dir * 1e-3, current_dir, RADIUS)
        opacity = max(0.42, 1.0 - reflections_done * 0.18)
        inside_segments.append((current_start, hit, opacity, color, 4.6))

        inward_normal = -normalize(hit - CENTER)
        if reflections_done == target_exit_reflections:
            exit_dir = refract(current_dir, inward_normal, n_water, AIR_REFRACTIVE_INDEX)
            if exit_dir is not None:
                exit_segments.append((hit, hit + exit_dir * OUT_LENGTH, 1.0, color, 5.0))
                preferred = {
                    "point": hit,
                    "dir": exit_dir,
                    "angle_deg": angle_between(exit_dir, LEFT),
                    "y_offset": y,
                }
            break

        reflected_dir = reflect(current_dir, inward_normal)
        reflection_segments.append((hit, hit + reflected_dir * 0.56, max(0.22, 0.75 - 0.12 * reflections_done), color, 2.6))
        current_start = hit
        current_dir = reflected_dir
        reflections_done += 1

    return {
        "color": color,
        "incoming": incoming_segments,
        "inside": inside_segments,
        "reflections": reflection_segments,
        "exits": exit_segments,
        "preferred": preferred,
        "entry": entry,
    }


def compute_angle_curve(wavelength_nm: float, target_exit_reflections: int = 1, n_steps: int = 90):
    points = []
    for i in range(n_steps + 1):
        y = Y_MAX * i / n_steps
        data = trace_ray_bundle(wavelength_nm, y, target_exit_reflections=target_exit_reflections)
        if data["preferred"] is not None:
            points.append((y, data["preferred"]["angle_deg"]))
    return points


def find_stationary_exit(wavelength_nm: float, target_exit_reflections: int = 1):
    samples = compute_angle_curve(wavelength_nm, target_exit_reflections=target_exit_reflections, n_steps=120)
    samples = [sample for sample in samples if sample[0] > 0.05]
    if not samples:
        return 0.8, 0.0
    angles = np.array([a for _, a in samples])
    idx = int(np.argmax(angles) if target_exit_reflections == 1 else np.argmin(angles))
    return samples[idx]


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


def build_beam_group(wavelength_nm: float, y_offset: float, target_exit_reflections: int = 1):
    bundle = trace_ray_bundle(wavelength_nm, y_offset, target_exit_reflections=target_exit_reflections)
    group = VGroup()
    for segment_group in (bundle["incoming"], bundle["inside"], bundle["exits"], bundle["reflections"]):
        for start, end, opacity, color, width in segment_group:
            group.add(glow_line(start, end, color, base_width=width, opacity=opacity, z_index=6 if width >= 4.0 else 5))
    return group


def build_exit_rays_only(wavelength_nm: float, y_values, target_exit_reflections: int = 1, opacity=0.26):
    group = VGroup()
    color = wavelength_to_rgb(wavelength_nm)
    for y in y_values:
        data = trace_ray_bundle(wavelength_nm, float(y), target_exit_reflections=target_exit_reflections)
        if data["preferred"] is None:
            continue
        start = data["preferred"]["point"]
        end = start + normalize(data["preferred"]["dir"]) * OUT_LENGTH
        group.add(glow_line(start, end, color, base_width=3.8, opacity=opacity, z_index=4))
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


def rainbow_arc(radius=2.6, center=ORIGIN, reverse=False, stroke_width=12):
    colors = [wavelength_to_rgb(650), wavelength_to_rgb(600), wavelength_to_rgb(570), wavelength_to_rgb(540), wavelength_to_rgb(490), wavelength_to_rgb(450)]
    if reverse:
        colors = list(reversed(colors))
    arcs = VGroup()
    for i, color in enumerate(colors):
        arc = Arc(radius=radius - i * 0.11, start_angle=PI, angle=PI, color=color, stroke_width=stroke_width)
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


def note_box(title, body, color=PRIMARY, width=4.5, height=1.9):
    box = RoundedRectangle(corner_radius=0.18, width=width, height=height, color=color)
    box.set_fill("#0D1620", opacity=0.95)
    box.set_stroke(color, width=2, opacity=0.8)
    head = Text(title, font=MONO, font_size=20, color=color, weight=BOLD)
    body_text = Text(body, font=MONO, font_size=17, color=TEXT, line_spacing=0.92)
    fit_text(body_text, width - 0.4)
    content = VGroup(head, body_text).arrange(DOWN, aligned_edge=LEFT, buff=0.16)
    content.move_to(box.get_center())
    content.align_to(box, LEFT).shift(RIGHT * 0.22)
    return VGroup(box, content)


def bullet_lines(lines, font_size=20, color=TEXT, buff=0.24):
    items = [Text(f"• {line}", font=MONO, font_size=font_size, color=color) for line in lines]
    return VGroup(*items).arrange(DOWN, aligned_edge=LEFT, buff=buff)


def add_title(scene, title, subtitle=None, accent=PRIMARY):
    head = Text(title, font=MONO, font_size=34, color=TEXT, weight=BOLD)
    head.to_edge(UP, buff=0.42)
    line = Line(LEFT * 6.2, RIGHT * 6.2, color=accent, stroke_width=2.2, stroke_opacity=0.42)
    line.next_to(head, DOWN, buff=0.18)
    scene.play(FadeIn(head, shift=UP * 0.15), run_time=0.7)
    scene.play(Create(line), run_time=0.5)
    sub = None
    if subtitle:
        sub = Text(subtitle, font=MONO, font_size=18, color=MUTED)
        fit_text(sub, 10.8)
        sub.next_to(line, DOWN, buff=0.18)
        scene.play(FadeIn(sub, shift=UP * 0.1), run_time=0.5)
    scene.wait(0.4)
    return VGroup(*[obj for obj in (head, line, sub) if obj is not None])


def clean_exit(scene):
    if scene.mobjects:
        scene.play(FadeOut(Group(*scene.mobjects)), run_time=0.6)
        scene.wait(0.2)


class AngleHook(Scene):
    def construct(self):
        add_title(
            self,
            "Nicht 'Sonne + Regen' – entscheidend ist der Winkel",
            "Der primäre Regenbogen erscheint nur aus ganz bestimmten Richtungen.",
            accent=ACCENT,
        )

        observer = build_eye(1.1).move_to(RIGHT * 4.8 + DOWN * 0.6)
        observer_label = Text("Beobachter", font=MONO, font_size=18, color=MUTED).next_to(observer, DOWN, buff=0.18)

        sun = Circle(radius=0.36, color=SUN, stroke_width=0).set_fill(SUN, opacity=1).move_to(RIGHT * 5.6 + UP * 2.3)
        sun_rays = VGroup(*[
            Line(sun.get_center() + rotate_vector(RIGHT * 0.48, ang), sun.get_center() + rotate_vector(RIGHT * 0.78, ang), color=SUN, stroke_width=2.5)
            for ang in np.linspace(0, TAU, 10, endpoint=False)
        ])
        sun_text = Text("Sonne im Rücken", font=MONO, font_size=18, color=SUN).next_to(sun, LEFT, buff=0.2)

        axis = DashedLine(observer.get_center(), observer.get_center() + LEFT * 5.2, dash_length=0.12, color=SOFT, stroke_width=1.8)
        axis_label = Text("Gegenpunkt der Sonne", font=MONO, font_size=18, color=SOFT).next_to(axis, DOWN, buff=0.18)

        red_dir = rotate_vector(LEFT, 42 * DEGREES)
        blue_dir = rotate_vector(LEFT, 40 * DEGREES)
        red_ray = Line(observer.get_center(), observer.get_center() + red_dir * 5.0, color=wavelength_to_rgb(650), stroke_width=5)
        blue_ray = Line(observer.get_center(), observer.get_center() + blue_dir * 4.7, color=wavelength_to_rgb(450), stroke_width=5)

        red_drops = VGroup(*[
            Circle(radius=0.12, color=wavelength_to_rgb(650), stroke_width=1.5).set_fill(wavelength_to_rgb(650), opacity=0.15).move_to(observer.get_center() + red_dir * d)
            for d in [2.2, 3.1, 4.0, 4.8]
        ])
        blue_drops = VGroup(*[
            Circle(radius=0.11, color=wavelength_to_rgb(450), stroke_width=1.5).set_fill(wavelength_to_rgb(450), opacity=0.15).move_to(observer.get_center() + blue_dir * d)
            for d in [2.1, 2.9, 3.7, 4.5]
        ])

        left_helper = Line(observer.get_center(), observer.get_center() + LEFT * 1.1, stroke_opacity=0)
        red_helper = Line(observer.get_center(), observer.get_center() + red_dir * 1.1, stroke_opacity=0)
        blue_helper = Line(observer.get_center(), observer.get_center() + blue_dir * 1.0, stroke_opacity=0)
        red_angle = Angle(left_helper, red_helper, radius=0.62, color=wavelength_to_rgb(650), stroke_width=3)
        blue_angle = Angle(left_helper, blue_helper, radius=0.42, color=wavelength_to_rgb(450), stroke_width=3)
        red_label = Text("≈ 42°", font=MONO, font_size=18, color=wavelength_to_rgb(650)).move_to(observer.get_center() + LEFT * 0.95 + UP * 0.7)
        blue_label = Text("≈ 40°", font=MONO, font_size=18, color=wavelength_to_rgb(450)).move_to(observer.get_center() + LEFT * 0.92 + UP * 0.45)

        bullets = bullet_lines([
            "Sonne steht hinter dem Beobachter.",
            "Rot kommt aus etwa 42°, Blau/Violett aus etwa 40°.",
            "Ein Regenbogen ist deshalb eine Winkelerscheinung.",
        ], font_size=19, color=MUTED)
        bullets.to_edge(LEFT, buff=0.55).shift(DOWN * 1.55)

        self.play(FadeIn(observer), FadeIn(observer_label), FadeIn(sun), FadeIn(sun_rays), FadeIn(sun_text), run_time=1.0)
        self.play(Create(axis), FadeIn(axis_label), run_time=0.8)
        self.wait(0.4)
        self.play(Create(red_ray), Create(blue_ray), run_time=1.1)
        self.play(FadeIn(red_drops, lag_ratio=0.1), FadeIn(blue_drops, lag_ratio=0.1), run_time=0.9)
        self.play(Create(red_angle), Create(blue_angle), FadeIn(red_label), FadeIn(blue_label), run_time=0.8)
        for bullet in bullets:
            self.play(FadeIn(bullet, shift=RIGHT * 0.12), run_time=0.35)
            self.wait(0.25)
        self.wait(1.8)
        clean_exit(self)


class RefractionAndReflection(Scene):
    def construct(self):
        add_title(
            self,
            "Was im Tropfen wirklich passiert",
            "Brechung lenkt das Licht um, Reflexion hält es im Tropfen.",
            accent=PRIMARY,
        )

        left_panel = RoundedRectangle(width=5.6, height=4.7, corner_radius=0.18, color=PRIMARY).set_fill("#0B1118", opacity=0.92).move_to(LEFT * 3.4 + DOWN * 0.35)
        right_panel = RoundedRectangle(width=5.6, height=4.7, corner_radius=0.18, color=ACCENT).set_fill("#0B1118", opacity=0.92).move_to(RIGHT * 3.4 + DOWN * 0.35)
        left_title = Text("Brechung", font=MONO, font_size=24, color=PRIMARY, weight=BOLD).next_to(left_panel, UP, buff=-0.45)
        right_title = Text("Innere Reflexion", font=MONO, font_size=24, color=ACCENT, weight=BOLD).next_to(right_panel, UP, buff=-0.45)

        interface_l = Line(LEFT * 5.35 + DOWN * 0.2, LEFT * 1.45 + DOWN * 0.2, color=GREY_B, stroke_width=2.5)
        normal_l = DashedLine(LEFT * 3.4 + UP * 1.8, LEFT * 3.4 + DOWN * 2.0, color=GREY_C, dash_length=0.12)
        air = Text("Luft", font=MONO, font_size=18, color=SOFT).move_to(LEFT * 5.0 + UP * 0.75)
        water = Text("Wasser", font=MONO, font_size=18, color=PRIMARY).move_to(LEFT * 5.0 + DOWN * 1.0)
        inc = Arrow(LEFT * 4.8 + UP * 1.55, LEFT * 3.42 + DOWN * 0.18, buff=0, color=YELLOW_B, stroke_width=3.4, max_tip_length_to_length_ratio=0.08)
        refr = Arrow(LEFT * 3.38 + DOWN * 0.22, LEFT * 2.55 + DOWN * 1.75, buff=0, color=PRIMARY, stroke_width=3.4, max_tip_length_to_length_ratio=0.08)
        left_formula = Text("n₂ > n₁  ⇒  θ₂ < θ₁", font=MONO, font_size=20, color=TEXT).move_to(LEFT * 3.4 + DOWN * 2.2)

        interface_r = Line(RIGHT * 1.55 + DOWN * 0.4, RIGHT * 5.25 + DOWN * 0.4, color=GREY_B, stroke_width=2.5)
        normal_r = DashedLine(RIGHT * 3.4 + UP * 1.6, RIGHT * 3.4 + DOWN * 2.0, color=GREY_C, dash_length=0.12)
        inc_r = Arrow(RIGHT * 2.0 + DOWN * 1.7, RIGHT * 3.38 + DOWN * 0.42, buff=0, color=GREEN_B, stroke_width=3.4, max_tip_length_to_length_ratio=0.08)
        refl_r = Arrow(RIGHT * 3.42 + DOWN * 0.42, RIGHT * 4.8 + DOWN * 1.7, buff=0, color=ACCENT, stroke_width=3.4, max_tip_length_to_length_ratio=0.08)
        rule_text = Text("θein = θaus", font=MONO, font_size=20, color=TEXT).move_to(RIGHT * 3.4 + DOWN * 2.2)
        wall_text = Text("an der Rückseite des Tropfens", font=MONO, font_size=17, color=MUTED).move_to(RIGHT * 3.4 + UP * 1.8)

        bridge = note_box(
            "Merke",
            "Erst Brechung + dann eine oder mehrere innere Reflexionen erzeugen die austretenden Farbstrahlen.",
            color=SECONDARY,
            width=10.5,
            height=1.55,
        ).move_to(DOWN * 3.0)

        self.play(FadeIn(left_panel), FadeIn(right_panel), FadeIn(left_title), FadeIn(right_title), run_time=0.8)
        self.play(Create(interface_l), Create(interface_r), Create(normal_l), Create(normal_r), run_time=0.8)
        self.play(FadeIn(air), FadeIn(water), FadeIn(wall_text), run_time=0.5)
        self.play(GrowArrow(inc), run_time=0.7)
        self.play(GrowArrow(refr), FadeIn(left_formula), run_time=0.8)
        self.wait(0.5)
        self.play(GrowArrow(inc_r), run_time=0.7)
        self.play(GrowArrow(refl_r), FadeIn(rule_text), run_time=0.8)
        self.wait(0.6)
        self.play(FadeIn(bridge), run_time=0.8)
        self.wait(2.0)
        clean_exit(self)


class SingleDropPhysics(Scene):
    def construct(self):
        add_title(
            self,
            "Ein Tropfen selektiert Auslenkungswinkel",
            "Die Farben verlassen den Tropfen nicht gleich, sondern mit leicht verschiedenen Winkeln.",
            accent=SECONDARY,
        )

        droplet = build_droplet()
        self.play(FadeIn(droplet), run_time=0.9)

        green_y, _ = find_stationary_exit(540.0, 1)
        incoming = trace_ray_bundle(540.0, green_y, target_exit_reflections=1)["incoming"][0]
        white_beam = glow_line(incoming[0], incoming[1], WHITE, base_width=5.4, opacity=1.0, z_index=6)
        white_label = Text("einfallendes Sonnenlicht", font=MONO, font_size=18, color=TEXT).move_to(scene_point(np.array([-5.1, green_y + 0.65, 0.0])))

        red_y, red_angle = find_stationary_exit(650.0, 1)
        green_y, green_angle = find_stationary_exit(540.0, 1)
        blue_y, blue_angle = find_stationary_exit(450.0, 1)

        red_group = build_beam_group(650.0, red_y, target_exit_reflections=1)
        green_group = build_beam_group(540.0, green_y, target_exit_reflections=1)
        blue_group = build_beam_group(450.0, blue_y, target_exit_reflections=1)

        red_data = trace_ray_bundle(650.0, red_y, target_exit_reflections=1)["preferred"]
        green_data = trace_ray_bundle(540.0, green_y, target_exit_reflections=1)["preferred"]
        blue_data = trace_ray_bundle(450.0, blue_y, target_exit_reflections=1)["preferred"]

        red_tag = Text(f"Rot  ≈ {red_angle:.1f}°", font=MONO, font_size=18, color=wavelength_to_rgb(650)).move_to(scene_point(red_data["point"] + normalize(red_data["dir"]) * 2.45 + np.array([0.0, 0.55, 0.0])))
        green_tag = Text(f"Grün ≈ {green_angle:.1f}°", font=MONO, font_size=18, color=wavelength_to_rgb(540)).move_to(scene_point(green_data["point"] + normalize(green_data["dir"]) * 2.4 + np.array([0.25, 0.0, 0.0])))
        blue_tag = Text(f"Blau ≈ {blue_angle:.1f}°", font=MONO, font_size=18, color=wavelength_to_rgb(450)).move_to(scene_point(blue_data["point"] + normalize(blue_data["dir"]) * 2.1 + np.array([0.1, -0.45, 0.0])))

        fan_red = build_exit_rays_only(650.0, np.linspace(max(0.0, red_y - 0.16), min(Y_MAX, red_y + 0.16), 9), opacity=0.22)
        fan_green = build_exit_rays_only(540.0, np.linspace(max(0.0, green_y - 0.16), min(Y_MAX, green_y + 0.16), 9), opacity=0.22)
        fan_blue = build_exit_rays_only(450.0, np.linspace(max(0.0, blue_y - 0.16), min(Y_MAX, blue_y + 0.16), 9), opacity=0.22)

        insight = note_box(
            "Aha",
            "Benachbarte Strahlen verlassen den Tropfen in fast derselben Richtung. Genau dort wird der Regenbogen hell.",
            color=ACCENT,
            width=4.9,
            height=2.1,
        ).to_edge(RIGHT, buff=0.45).shift(DOWN * 0.25)

        self.play(FadeIn(white_beam), FadeIn(white_label), run_time=0.8)
        self.wait(0.4)
        self.play(FadeIn(red_group), FadeIn(green_group), FadeIn(blue_group), run_time=1.2)
        self.play(FadeIn(red_tag), FadeIn(green_tag), FadeIn(blue_tag), run_time=0.8)
        self.wait(0.6)
        self.play(FadeIn(fan_red), FadeIn(fan_green), FadeIn(fan_blue), run_time=0.9)
        self.play(FadeIn(insight), run_time=0.8)
        self.wait(2.0)
        clean_exit(self)


class AngleSelectionGraph(Scene):
    def construct(self):
        add_title(
            self,
            "Warum gerade 40° bis 42°?",
            "Der Auslenkungswinkel besitzt ein Extremum – dort stauen sich die Strahlen.",
            accent=PRIMARY,
        )

        all_samples = [(name, wl, compute_angle_curve(wl, target_exit_reflections=1, n_steps=120)) for name, wl in SELECTED_COLORS]
        y_values = [angle for _, _, pts in all_samples for _, angle in pts]
        y_min = np.floor(min(y_values) - 0.4)
        y_max = np.ceil(max(y_values) + 0.4)

        axes = Axes(
            x_range=[0, Y_MAX, round(Y_MAX / 4, 2)],
            y_range=[y_min, y_max, 1],
            x_length=8.8,
            y_length=5.2,
            axis_config={
                "color": GREY_B,
                "stroke_opacity": 0.72,
                "stroke_width": 2.0,
                "include_numbers": False,
            },
            tips=False,
        ).move_to(LEFT * 1.2 + DOWN * 0.3)

        x_label = Text("Einfallshöhe y", font=MONO, font_size=22, color=MUTED).next_to(axes, DOWN, buff=0.36)
        y_label = Text("Ablenkwinkel [°]", font=MONO, font_size=22, color=MUTED).rotate(PI / 2).next_to(axes, LEFT, buff=0.45)

        x_ticks = VGroup(*[
            Text(label, font=MONO, font_size=16, color=SOFT).next_to(axes.c2p(x, y_min), DOWN, buff=0.15)
            for x, label in [(0.0, "0"), (round(Y_MAX / 2, 1), "R/2"), (round(Y_MAX, 1), "R")]
        ])
        y_ticks = VGroup(*[
            Text(f"{int(val)}°", font=MONO, font_size=16, color=SOFT).next_to(axes.c2p(0, val), LEFT, buff=0.14)
            for val in [40, 41, 42]
            if y_min <= val <= y_max
        ])

        self.play(FadeIn(axes), FadeIn(x_label), FadeIn(y_label), FadeIn(x_ticks), FadeIn(y_ticks), run_time=0.8)

        curves = VGroup()
        labels = VGroup()
        markers = VGroup()
        for name, wl, pts in all_samples:
            color = wavelength_to_rgb(wl)
            curve_points = [axes.c2p(x, y) for x, y in pts]
            glow = VMobject(color=color)
            glow.set_points_as_corners(curve_points)
            glow.set_stroke(color=color, width=12, opacity=0.12)
            curve = VMobject(color=color)
            curve.set_points_as_corners(curve_points)
            curve.set_stroke(color=color, width=4.8)
            curves.add(glow, curve)

            station_y, station_angle = find_stationary_exit(wl, 1)
            station_dot = Dot(axes.c2p(station_y, station_angle), radius=0.06, color=color)
            station_label = Text(f"{name}: {station_angle:.1f}°", font=MONO, font_size=18, color=color).next_to(station_dot, RIGHT, buff=0.16)
            markers.add(station_dot, station_label)

            labels.add(Text(f"{name} {int(wl)} nm", font=MONO, font_size=18, color=color))

        labels.arrange(DOWN, aligned_edge=LEFT, buff=0.22).to_edge(RIGHT, buff=0.45).shift(UP * 0.8)

        for i in range(0, len(curves), 2):
            self.play(Create(curves[i + 1]), FadeIn(curves[i]), run_time=1.0)
        self.play(FadeIn(labels), run_time=0.6)
        self.wait(0.5)

        for i in range(0, len(markers), 2):
            self.play(FadeIn(markers[i]), FadeIn(markers[i + 1]), run_time=0.45)

        insight = note_box(
            "Stationärer Winkel",
            "Am Extremum gilt näherungsweise: dD/dy = 0. Dadurch liefern viele benachbarte Strahlen fast denselben Beobachtungswinkel.",
            color=ACCENT,
            width=4.9,
            height=2.2,
        ).to_edge(RIGHT, buff=0.45).shift(DOWN * 1.6)
        self.play(FadeIn(insight), run_time=0.8)
        self.wait(2.3)
        clean_exit(self)


class WhyArcNotSpot(Scene):
    def construct(self):
        add_title(
            self,
            "Warum ein Bogen – und kein einzelner Punkt?",
            "Der Regenbogen ist ein Kreis um den Gegenpunkt der Sonne.",
            accent=ACCENT,
        )

        sky = Circle(radius=3.25, color=GREY_C, stroke_opacity=0.35).move_to(DOWN * 0.15)
        horizon = Line(LEFT * 5.8 + DOWN * 1.2, RIGHT * 5.8 + DOWN * 1.2, color=GREY_B, stroke_width=2.2)
        ground_fill = Rectangle(width=12.0, height=2.2, stroke_width=0).set_fill("#0A1118", opacity=1.0).move_to(DOWN * 2.25)

        observer = build_eye(0.9).move_to(DOWN * 1.8)
        observer_label = Text("Beobachter", font=MONO, font_size=18, color=MUTED).next_to(observer, DOWN, buff=0.18)
        sun = Circle(radius=0.24, color=SUN, stroke_width=0).set_fill(SUN, opacity=1).move_to(RIGHT * 4.5 + DOWN * 1.85)
        sun_text = Text("Sonne", font=MONO, font_size=17, color=SUN).next_to(sun, UP, buff=0.1)

        antisolar = Dot(point=DOWN * 2.25, radius=0.06, color=WHITE)
        antisolar_label = Text("Gegenpunkt der Sonne", font=MONO, font_size=18, color=SOFT).next_to(antisolar, DOWN, buff=0.18)

        primary_red = Circle(radius=2.55, color=wavelength_to_rgb(650), stroke_width=6).move_to(antisolar.get_center())
        primary_blue = Circle(radius=2.38, color=wavelength_to_rgb(450), stroke_width=6).move_to(antisolar.get_center())
        visible_arc = Arc(radius=2.55, start_angle=20 * DEGREES, angle=140 * DEGREES, color=wavelength_to_rgb(650), stroke_width=10).move_arc_center_to(antisolar.get_center())
        visible_arc_inner = Arc(radius=2.38, start_angle=20 * DEGREES, angle=140 * DEGREES, color=wavelength_to_rgb(450), stroke_width=10).move_arc_center_to(antisolar.get_center())

        bullets = bullet_lines([
            "Im Prinzip ist der Regenbogen ein ganzer Kreis.",
            "Vom Boden aus verdeckt der Horizont die untere Hälfte.",
            "Aus Flugzeug oder auf einem Berg kann mehr vom Kreis sichtbar werden.",
        ], font_size=19, color=MUTED)
        bullets.to_edge(RIGHT, buff=0.4).shift(DOWN * 0.45)

        self.play(FadeIn(sky), FadeIn(ground_fill), Create(horizon), run_time=0.8)
        self.play(FadeIn(observer), FadeIn(observer_label), FadeIn(sun), FadeIn(sun_text), run_time=0.7)
        self.play(FadeIn(antisolar), FadeIn(antisolar_label), run_time=0.6)
        self.play(Create(primary_red), Create(primary_blue), run_time=1.0)
        self.wait(0.5)
        self.play(Transform(primary_red, visible_arc), Transform(primary_blue, visible_arc_inner), run_time=1.0)
        for bullet in bullets:
            self.play(FadeIn(bullet, shift=LEFT * 0.12), run_time=0.35)
            self.wait(0.25)
        self.wait(2.0)
        clean_exit(self)


class PrimarySecondaryRainbow(Scene):
    def construct(self):
        add_title(
            self,
            "Primärer und sekundärer Regenbogen",
            "Mehr innere Reflexionen bedeuten anderen Winkel, schwächere Intensität und vertauschte Farbfolge.",
            accent=SECONDARY,
        )

        left_drop = build_droplet().scale(0.72).move_to(LEFT * 3.7 + DOWN * 0.2)
        right_drop = build_droplet().scale(0.72).move_to(RIGHT * 1.4 + DOWN * 0.2)

        primary_y, primary_angle = find_stationary_exit(650.0, 1)
        secondary_y, secondary_angle = find_stationary_exit(650.0, 2)

        primary_red = build_beam_group(650.0, primary_y, target_exit_reflections=1).scale(0.72).move_to(left_drop.get_center() - SCENE_SHIFT * 0.28)
        primary_blue = build_beam_group(450.0, find_stationary_exit(450.0, 1)[0], target_exit_reflections=1).scale(0.72).move_to(left_drop.get_center() - SCENE_SHIFT * 0.28)
        secondary_red = build_beam_group(650.0, secondary_y, target_exit_reflections=2).scale(0.72).move_to(right_drop.get_center() - SCENE_SHIFT * 0.28)
        secondary_blue = build_beam_group(450.0, find_stationary_exit(450.0, 2)[0], target_exit_reflections=2).scale(0.72).move_to(right_drop.get_center() - SCENE_SHIFT * 0.28)

        primary_arc = rainbow_arc(radius=1.18, center=LEFT * 3.7 + UP * 2.0, reverse=False, stroke_width=10)
        secondary_arc = rainbow_arc(radius=1.18, center=RIGHT * 1.4 + UP * 2.0, reverse=True, stroke_width=10)

        primary_box = note_box(
            "Primär",
            f"1 innere Reflexion\nheller\nRot außen\n≈ {primary_angle:.1f}°",
            color=PRIMARY,
            width=2.9,
            height=2.45,
        ).to_edge(LEFT, buff=0.35).shift(DOWN * 1.75)

        secondary_box = note_box(
            "Sekundär",
            f"2 innere Reflexionen\nschwächer\nRot innen\n≈ {secondary_angle:.1f}°",
            color=ACCENT,
            width=3.15,
            height=2.45,
        ).to_edge(RIGHT, buff=0.35).shift(DOWN * 1.75)

        dark_band = RoundedRectangle(width=1.75, height=2.15, corner_radius=0.14, color=GREY_B)
        dark_band.set_fill(BLACK, opacity=0.22)
        dark_band.set_stroke(GREY_B, width=1.8, opacity=0.3)
        dark_band.move_to(LEFT * 1.15 + UP * 1.95)
        dark_text = Text("Alexander-\nBand", font=MONO, font_size=17, color=MUTED, line_spacing=1.0).move_to(dark_band.get_center())

        self.play(FadeIn(left_drop), FadeIn(right_drop), run_time=0.8)
        self.play(FadeIn(primary_arc), FadeIn(secondary_arc), run_time=0.8)
        self.play(FadeIn(primary_red), FadeIn(primary_blue), run_time=0.9)
        self.play(FadeIn(secondary_red), FadeIn(secondary_blue), run_time=0.9)
        self.play(FadeIn(primary_box), FadeIn(secondary_box), FadeIn(dark_band), FadeIn(dark_text), run_time=0.8)
        self.wait(2.3)
        clean_exit(self)


class FinalTakeaway(Scene):
    def construct(self):
        add_title(
            self,
            "Fazit auf Oberstufen-Niveau",
            "Der Regenbogen ist eine Richtungs- und Intensitätsstruktur des Lichts.",
            accent=PRIMARY,
        )

        cards = VGroup(
            note_box("1. Brechung", "Beim Eintritt und Austritt werden die Farben verschieden stark abgelenkt.", color=PRIMARY, width=3.6, height=2.3),
            note_box("2. Innere Reflexion", "Erst Reflexionen im Tropfen erzeugen die primären und sekundären Wege.", color=ACCENT, width=3.6, height=2.3),
            note_box("3. Stationärer Winkel", "Am Extremum des Auslenkungswinkels sammelt sich die Helligkeit.", color=SECONDARY, width=3.6, height=2.3),
        ).arrange(RIGHT, buff=0.32).move_to(UP * 0.55)

        final_line = Text(
            "Ein Regenbogen ist kein Objekt an einem Ort –\nsondern Licht aus einer ganz bestimmten Richtung.",
            font=MONO,
            font_size=26,
            color=TEXT,
            weight=BOLD,
            line_spacing=0.95,
        ).move_to(DOWN * 2.15)
        fit_text(final_line, 10.8)

        self.play(FadeIn(cards[0], shift=UP * 0.15), run_time=0.7)
        self.wait(0.4)
        self.play(FadeIn(cards[1], shift=UP * 0.15), run_time=0.7)
        self.wait(0.4)
        self.play(FadeIn(cards[2], shift=UP * 0.15), run_time=0.7)
        self.wait(0.6)
        self.play(Write(final_line), run_time=1.0)
        self.wait(2.0)
        clean_exit(self)
