from manim import *
import numpy as np

# Render command:
# manim -pql refraction_mechanism_v2.py RefractionMechanism
# manim -pqh refraction_mechanism_v2.py RefractionMechanism

config.background_color = "#0C0F14"

MONO = "DejaVu Sans Mono"

# Palette
PRIMARY   = "#58C4DD"
ACCENT    = "#FFD166"
SECONDARY = "#83C167"
TEXT      = "#F5F7FA"
SOFT      = "#7F8A99"
SUN       = "#FFD54F"
RED_WL    = "#FF4444"
PANEL     = "#111820"

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def caption(text, color=TEXT, font_size=26):
    label = Text(text, font=MONO, font_size=font_size, color=color)
    box = RoundedRectangle(corner_radius=0.12, width=label.width + 0.5,
                           height=label.height + 0.28, color=color)
    box.set_fill(PANEL, opacity=0.88)
    box.set_stroke(color, width=1.4, opacity=0.65)
    box.move_to(label)
    return VGroup(box, label).to_edge(DOWN, buff=0.28)

def mk_wave(func, x0, x1, color, width=3.5, n=300):
    xs = np.linspace(x0, x1, n)
    pts = [[x, func(x), 0] for x in xs]
    c = VMobject(color=color, stroke_width=width)
    c.set_points_as_corners(pts)
    return c

def section_title(text, color=ACCENT):
    label = Text(text, font=MONO, font_size=26, color=color)
    box = RoundedRectangle(corner_radius=0.13, width=label.width + 0.55,
                           height=label.height + 0.28, color=color)
    box.set_fill(PANEL, opacity=0.90)
    box.set_stroke(color, width=1.5, opacity=0.78)
    box.move_to(label)
    return VGroup(box, label).to_edge(UP, buff=0.20)


# ──────────────────────────────────────────────────────────────────────────────
# Main scene
# ──────────────────────────────────────────────────────────────────────────────

class RefractionMechanism(Scene):
    """
    A step-by-step animation explaining WHY light refracts.
    Structured in 6 sub-scenes that build on each other.
    """

    def construct(self):
        self.camera.background_color = "#0C0F14"

        self._sub_00_hook()
        self._sub_01_single_atom_wave()
        self._sub_02_phase_lag_detail()
        self._sub_03_superposition()
        self._sub_04_slowdown()
        self._sub_05_collective_snell()

    # ── 0 · Hook ─────────────────────────────────────────────────────────────

    def _sub_00_hook(self):
        q = Text("Warum bricht Licht?", font=MONO, font_size=52, color=TEXT)
        sub = Text(
            "Weil Materie Licht verlangsamt — aber warum?",
            font=MONO, font_size=24, color=SOFT,
        ).next_to(q, DOWN, buff=0.42)
        self.play(FadeIn(q, shift=UP * 0.12), run_time=0.8)
        self.play(FadeIn(sub), run_time=0.6)
        self.wait(1.8)
        self.play(FadeOut(VGroup(q, sub)), run_time=0.5)

    # ── 1 · Einzelatom trifft Welle ──────────────────────────────────────────

    def _sub_01_single_atom_wave(self):
        title = section_title("Schritt 1: Licht als elektromagnetische Welle")
        self.play(FadeIn(title), run_time=0.4)
        self.wait(0.25)

        # ── static background: E-field arrow legend ──
        e_label = Text("E", font=MONO, font_size=22, color=PRIMARY)
        e_arrow = Arrow(ORIGIN, UP * 0.6, color=PRIMARY, stroke_width=3,
                        max_tip_length_to_length_ratio=0.25)
        e_grp = VGroup(e_arrow, e_label.next_to(e_arrow, RIGHT, buff=0.08)).to_corner(UL, buff=0.55)
        e_grp.shift(DOWN * 0.8)

        prop_arrow = Arrow(LEFT * 1.0, RIGHT * 1.0, color=ACCENT, stroke_width=2.5,
                           max_tip_length_to_length_ratio=0.18)
        prop_label = Text("v", font=MONO, font_size=20, color=ACCENT)
        prop_grp = VGroup(prop_arrow, prop_label.next_to(prop_arrow, UP, buff=0.06))
        prop_grp.to_corner(UR, buff=0.75).shift(DOWN * 0.8)

        # ── the atom ──
        nucleus = Dot(radius=0.22, color=ACCENT)
        nucleus_glow = Circle(radius=0.38, color=ACCENT, stroke_opacity=0.2, stroke_width=8)
        atom_label = Text("Kern", font=MONO, font_size=18, color=SOFT).next_to(nucleus, DOWN, buff=0.3)

        electron = Dot(radius=0.11, color=SUN)
        e_orbit = Circle(radius=0.55, color=SOFT, stroke_opacity=0.32, stroke_width=1.5)
        e_orbit.move_to(nucleus)
        electron.move_to(nucleus.get_center() + RIGHT * 0.55)
        e_orbit_label = Text("e⁻", font=MONO, font_size=18, color=SUN).next_to(electron, UP, buff=0.08)

        atom_grp = VGroup(nucleus_glow, e_orbit, nucleus, electron)
        atom_grp.move_to(ORIGIN)
        atom_label.move_to(nucleus.get_center() + DOWN * 0.52)

        # Wave parameters
        k = 2 * PI / 2.8   # spatial frequency
        omega = 2 * PI / 2.8  # temporal frequency (normalized — 1 unit time = 1 period)

        t = self.renderer.time  # live time reference

        # Left incoming wave (vacuum, x ∈ [-7, atom_x])
        ATOM_X = 0.0
        X_LEFT = -7.0

        def incident_func(x, time):
            return 0.9 * np.sin(k * x - omega * time)

        incoming_wave = always_redraw(lambda: mk_wave(
            lambda x: incident_func(x, self.renderer.time),
            X_LEFT, ATOM_X - 0.22,
            color=PRIMARY, width=4.0,
        ))

        cap = caption("Licht ist eine Welle aus oszillierenden E- und B-Feldern.")

        self.play(FadeIn(e_grp), FadeIn(prop_grp), run_time=0.6)
        self.play(FadeIn(atom_grp), Write(atom_label), FadeIn(e_orbit_label), run_time=0.75)
        self.play(Create(incoming_wave), run_time=0.8)
        self.play(FadeIn(cap), run_time=0.4)
        self.wait(3.5)

        self.play(FadeOut(VGroup(
            title, e_grp, prop_grp, atom_grp, atom_label,
            e_orbit_label, incoming_wave, cap
        )), run_time=0.5)

    # ── 2 · Phasenverzögerung detailliert ────────────────────────────────────

    def _sub_02_phase_lag_detail(self):
        title = section_title("Schritt 2: Erzwungene Schwingung & Phasenlag")
        self.play(FadeIn(title), run_time=0.4)

        k = 2 * PI / 2.8
        omega = 2 * PI / 2.8
        PHASE_LAG = PI / 2   # 90° — classic resonance lag

        ATOM_X = 0.5

        # ── Split layout: left = diagram, right = phase-wheel ──

        # --- Diagram side (left) ---
        divider = DashedLine(UP * 3.6 + RIGHT * 0.2, DOWN * 3.6 + RIGHT * 0.2,
                             dash_length=0.10, color=SOFT, stroke_width=1.2, stroke_opacity=0.45)

        atom_dot = Dot(radius=0.22, color=ACCENT).move_to(LEFT * 3.0)
        atom_ring = Circle(radius=0.44, color=ACCENT, stroke_opacity=0.22, stroke_width=10)
        atom_ring.move_to(atom_dot)
        elec = Dot(radius=0.10, color=SUN)

        ATOM_SCENE_X = atom_dot.get_x()  # -3.0

        def incident_func(x, time):
            return 0.75 * np.sin(k * x - omega * time)

        def secondary_func(x, time):
            # secondary wave emitted by electron: same k, but phase-shifted
            return 0.48 * np.sin(k * x - omega * time - PHASE_LAG)

        # Electron follows incident field but lags
        def elec_y(time):
            return incident_func(ATOM_SCENE_X, time - PHASE_LAG / omega) * 0.62

        elec.add_updater(lambda m: m.move_to(atom_dot.get_center() + UP * elec_y(self.renderer.time)))

        inc_wave = always_redraw(lambda: mk_wave(
            lambda x: incident_func(x, self.renderer.time),
            -7.0, ATOM_SCENE_X - 0.22, PRIMARY, width=4.2,
        ))
        sec_wave = always_redraw(lambda: mk_wave(
            lambda x: secondary_func(x, self.renderer.time),
            ATOM_SCENE_X + 0.22, 0.2, SECONDARY, width=3.4,
        ))

        # E-field at atom position (dashed vertical oscillator)
        e_indicator = always_redraw(lambda: Arrow(
            atom_dot.get_center(),
            atom_dot.get_center() + UP * incident_func(ATOM_SCENE_X, self.renderer.time) * 1.1,
            color=PRIMARY, buff=0, stroke_width=2.8,
            max_tip_length_to_length_ratio=0.18
        ))

        inc_label = Text("Einfallendes E-Feld", font=MONO, font_size=18, color=PRIMARY).to_corner(UL, buff=0.55).shift(DOWN * 0.6)
        sec_label = Text("Sekundärwelle (e⁻ → Welle)", font=MONO, font_size=18, color=SECONDARY).next_to(inc_label, DOWN, aligned_edge=LEFT, buff=0.15)

        # --- Phase wheel (right side) ---
        wheel_center = RIGHT * 3.5 + DOWN * 0.3
        wheel_radius = 1.05

        wheel_circle = Circle(radius=wheel_radius, color=SOFT, stroke_opacity=0.5, stroke_width=1.8)
        wheel_circle.move_to(wheel_center)

        # Reference phasor (incident)
        inc_phasor = always_redraw(lambda: Arrow(
            wheel_center,
            wheel_center + np.array([
                np.cos(-omega * self.renderer.time),
                np.sin(-omega * self.renderer.time),
                0
            ]) * wheel_radius,
            color=PRIMARY, buff=0, stroke_width=3.5,
            max_tip_length_to_length_ratio=0.20
        ))

        # Lagging phasor (secondary / electron)
        sec_phasor = always_redraw(lambda: Arrow(
            wheel_center,
            wheel_center + np.array([
                np.cos(-omega * self.renderer.time - PHASE_LAG),
                np.sin(-omega * self.renderer.time - PHASE_LAG),
                0
            ]) * wheel_radius,
            color=SECONDARY, buff=0, stroke_width=3.5,
            max_tip_length_to_length_ratio=0.20
        ))

        # Arc showing the lag angle
        lag_arc = always_redraw(lambda: Arc(
            radius=wheel_radius * 0.55,
            start_angle=-omega * self.renderer.time - PHASE_LAG,
            angle=PHASE_LAG,
            color=ACCENT, stroke_width=3.5,
            arc_center=wheel_center
        ))

        lag_label = Text("90° Lag", font=MONO, font_size=20, color=ACCENT)
        lag_label.next_to(wheel_circle, DOWN, buff=0.28)

        wheel_title = Text("Phasenzeiger", font=MONO, font_size=20, color=TEXT)
        wheel_title.next_to(wheel_circle, UP, buff=0.28)

        cap1 = caption("Das E-Feld des Lichts treibt das Elektron zur Schwingung —\nmit einer Phasenverzögerung (hier: 90°).")

        self.play(
            FadeIn(divider),
            FadeIn(atom_dot), FadeIn(atom_ring), FadeIn(elec),
            FadeIn(inc_label), FadeIn(sec_label),
            run_time=0.7
        )
        self.play(Create(inc_wave), FadeIn(e_indicator), run_time=0.7)
        self.play(FadeIn(wheel_circle), FadeIn(wheel_title),
                  FadeIn(inc_phasor), run_time=0.6)
        self.wait(1.2)

        self.play(Create(sec_wave), FadeIn(sec_phasor), FadeIn(lag_arc),
                  FadeIn(lag_label), run_time=0.8)
        self.play(FadeIn(cap1), run_time=0.4)
        self.wait(4.0)

        self.play(FadeOut(Group(*self.mobjects)), run_time=0.5)

    # ── 3 · Überlagerung = Wellengeschwindigkeit ──────────────────────────────

    def _sub_03_superposition(self):
        title = section_title("Schritt 3: Überlagerung verschiebt die Phase")
        self.play(FadeIn(title), run_time=0.4)

        k = 2 * PI / 2.8
        omega = 2 * PI / 2.8
        PHASE_LAG = PI / 2
        AMP_INC = 0.85
        AMP_SEC = 0.38   # smaller secondary contribution per atom

        ATOM_X = 0.0
        X_L = -6.5
        X_R = 6.5

        def inc(x, t):
            return AMP_INC * np.sin(k * x - omega * t)

        def sec(x, t):
            # Only right of atom
            if x < ATOM_X:
                return 0.0
            return AMP_SEC * np.sin(k * x - omega * t - PHASE_LAG)

        def result(x, t):
            return inc(x, t) + sec(x, t)

        # Waves
        inc_wave = always_redraw(lambda: mk_wave(
            lambda x: inc(x, self.renderer.time), X_L, X_R, PRIMARY, width=3.0
        ))
        sec_wave = always_redraw(lambda: mk_wave(
            lambda x: (AMP_SEC * np.sin(k * x - omega * self.renderer.time - PHASE_LAG)
                       if x >= ATOM_X else 0),
            X_L, X_R, SECONDARY, width=2.8
        ))
        result_wave = always_redraw(lambda: mk_wave(
            lambda x: result(x, self.renderer.time), X_L, X_R,
            color=ACCENT, width=5.0
        ))

        # Vertical "crest tracker" lines to show phase shift
        # Track the nearest crest of incident vs. result to the right of atom
        def crest_x_incident(t):
            # sin(kx - wt) = 1 → x = (PI/2 + wt) / k, choosing closest right of atom
            raw = (PI / 2 + omega * t) / k
            # bring to range [ATOM_X, ATOM_X + 2*PI/k]
            lam = 2 * PI / k
            return ATOM_X + ((raw - ATOM_X) % lam)

        def crest_x_result(t):
            # find numerically
            xs = np.linspace(ATOM_X + 0.05, ATOM_X + 2 * PI / k, 400)
            ys = np.array([result(x, t) for x in xs])
            idx = int(np.argmax(ys))
            return xs[idx]

        inc_crest_line = always_redraw(lambda: DashedLine(
            [crest_x_incident(self.renderer.time), -1.6, 0],
            [crest_x_incident(self.renderer.time),  1.6, 0],
            color=PRIMARY, stroke_width=1.8, dash_length=0.10, stroke_opacity=0.72
        ))
        result_crest_line = always_redraw(lambda: DashedLine(
            [crest_x_result(self.renderer.time), -1.6, 0],
            [crest_x_result(self.renderer.time),  1.6, 0],
            color=ACCENT, stroke_width=1.8, dash_length=0.10, stroke_opacity=0.72
        ))

        atom_dot = Dot(radius=0.18, color=WHITE, stroke_color=SOFT, stroke_width=2)
        atom_dot.move_to([ATOM_X, 0, 0])
        atom_mark = DashedLine([ATOM_X, -2.2, 0], [ATOM_X, 2.2, 0],
                               dash_length=0.08, color=SOFT, stroke_width=1.2, stroke_opacity=0.45)

        legend_inc = VGroup(
            Line(LEFT * 0.35, RIGHT * 0.35, color=PRIMARY, stroke_width=3),
            Text("Einfall", font=MONO, font_size=19, color=PRIMARY)
        ).arrange(RIGHT, buff=0.14)
        legend_sec = VGroup(
            Line(LEFT * 0.35, RIGHT * 0.35, color=SECONDARY, stroke_width=2.6),
            Text("Sekundär", font=MONO, font_size=19, color=SECONDARY)
        ).arrange(RIGHT, buff=0.14)
        legend_res = VGroup(
            Line(LEFT * 0.35, RIGHT * 0.35, color=ACCENT, stroke_width=4),
            Text("Summe", font=MONO, font_size=19, color=ACCENT)
        ).arrange(RIGHT, buff=0.14)
        legend = VGroup(legend_inc, legend_sec, legend_res).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
        legend.to_corner(UR, buff=0.52).shift(DOWN * 0.5)

        cap = caption("Hinter dem Atom verschiebt die Überlagerung die Wellenkämme nach hinten.\n→ Die Welle läuft scheinbar langsamer.")

        self.play(
            FadeIn(atom_mark), FadeIn(atom_dot),
            FadeIn(legend), run_time=0.7
        )
        self.play(Create(inc_wave), run_time=0.6)
        self.wait(0.5)
        self.play(Create(sec_wave), run_time=0.6)
        self.wait(0.5)
        self.play(
            inc_wave.animate.set_stroke(opacity=0.38),
            sec_wave.animate.set_stroke(opacity=0.38),
            Create(result_wave),
            Create(inc_crest_line),
            Create(result_crest_line),
            run_time=0.9
        )
        self.play(FadeIn(cap), run_time=0.5)
        self.wait(4.5)

        self.play(FadeOut(Group(*self.mobjects)), run_time=0.5)

    # ── 4 · Verlangsamung → Brechungsindex ───────────────────────────────────

    def _sub_04_slowdown(self):
        title = section_title("Schritt 4: Verlangsamung = Brechungsindex")
        self.play(FadeIn(title), run_time=0.4)

        # Side-by-side: vacuum vs medium wave trains
        k_vac = 2 * PI / 2.8
        k_med = 2 * PI / 2.0   # shorter wavelength inside medium (n = 1.4)
        omega  = 2 * PI / 2.8  # same frequency!

        BOUNDARY_X = 0.05
        X_L = -6.5
        X_R  = 6.5

        boundary = DashedLine([BOUNDARY_X, -3.3, 0], [BOUNDARY_X, 3.3, 0],
                              dash_length=0.10, color=SOFT, stroke_width=1.8, stroke_opacity=0.70)

        vac_label  = Text("Vakuum (c)", font=MONO, font_size=22, color=TEXT).move_to(LEFT * 3.2 + UP * 2.8)
        med_label  = Text("Medium (c/n)", font=MONO, font_size=22, color=TEXT).move_to(RIGHT * 3.2 + UP * 2.8)
        vac_bg = RoundedRectangle(corner_radius=0.10, width=6.3, height=5.8, color=SOFT)
        vac_bg.set_fill("#101520", opacity=0.55).set_stroke(SOFT, width=1.0, opacity=0.3)
        vac_bg.move_to(LEFT * 3.3)
        med_bg = RoundedRectangle(corner_radius=0.10, width=6.3, height=5.8, color=PRIMARY)
        med_bg.set_fill("#0E2035", opacity=0.55).set_stroke(PRIMARY, width=1.0, opacity=0.3)
        med_bg.move_to(RIGHT * 3.3)

        wave_vac = always_redraw(lambda: mk_wave(
            lambda x: 0.88 * np.sin(k_vac * x - omega * self.renderer.time),
            X_L, BOUNDARY_X, TEXT, width=4.2
        ))
        wave_med = always_redraw(lambda: mk_wave(
            lambda x: 0.88 * np.sin(k_med * x - omega * self.renderer.time),
            BOUNDARY_X, X_R, PRIMARY, width=4.2
        ))

        # Wavelength annotations
        lam_vac_brace_y = -1.45
        lam_med_brace_y = -1.45

        def lam_brace(side, color, y):
            # Show a double arrow spanning one wavelength
            if side == "vac":
                lam = 2 * PI / k_vac
                cx = -3.0
            else:
                lam = 2 * PI / k_med
                cx = 3.0
            left  = [cx - lam / 2, y, 0]
            right = [cx + lam / 2, y, 0]
            arr = DoubleArrow(left, right, color=color, stroke_width=2.4,
                              max_tip_length_to_length_ratio=0.12, tip_length=0.15)
            return arr

        arr_vac = lam_brace("vac", TEXT, lam_vac_brace_y)
        arr_med = lam_brace("med", PRIMARY, lam_med_brace_y)
        lam_vac_txt = MathTex(r"\lambda", color=TEXT, font_size=26).next_to(arr_vac, DOWN, buff=0.12)
        lam_med_txt = MathTex(r"\lambda/n", color=PRIMARY, font_size=26).next_to(arr_med, DOWN, buff=0.12)

        # Formula
        formula = MathTex(
            r"n = \frac{c}{v_\text{med}} = \frac{\lambda_\text{vac}}{\lambda_\text{med}}",
            color=ACCENT, font_size=38
        ).move_to(DOWN * 2.55)
        fbox = RoundedRectangle(corner_radius=0.14, width=formula.width + 0.5,
                                height=formula.height + 0.36, color=ACCENT)
        fbox.set_fill(PANEL, opacity=0.92).set_stroke(ACCENT, width=1.8, opacity=0.75)
        fbox.move_to(formula)

        cap = caption("Gleiche Frequenz, kürzere Wellenlänge → kleinere Phasengeschwindigkeit → n > 1")

        self.play(FadeIn(vac_bg), FadeIn(med_bg), run_time=0.6)
        self.play(Create(boundary), FadeIn(vac_label), FadeIn(med_label), run_time=0.6)
        self.play(Create(wave_vac), Create(wave_med), run_time=0.8)
        self.play(FadeIn(arr_vac), FadeIn(lam_vac_txt),
                  FadeIn(arr_med), FadeIn(lam_med_txt), run_time=0.7)
        self.play(FadeIn(fbox), Write(formula), run_time=0.8)
        self.play(FadeIn(cap), run_time=0.4)
        self.wait(4.2)

        self.play(FadeOut(Group(*self.mobjects)), run_time=0.5)

    # ── 5 · Kollektiv & Snell ────────────────────────────────────────────────

    def _sub_05_collective_snell(self):
        title = section_title("Schritt 5: Kollektiv → Snellsches Gesetz")
        self.play(FadeIn(title), run_time=0.4)

        k_vac = 2 * PI / 2.8
        k_med = 2 * PI / 2.0
        omega  = 2 * PI / 2.8

        # ── Wavefront diagram ──
        # Show a tilted plane wave hitting a flat medium surface
        # and bending (Huygens principle visual)

        SURF_Y = 0.25   # y of the surface (horizontal)

        surface = Line(LEFT * 6.2 + [0, SURF_Y, 0], RIGHT * 6.2 + [0, SURF_Y, 0],
                       color=PRIMARY, stroke_width=2.2, stroke_opacity=0.55)
        top_bg = Rectangle(width=14, height=4.0, stroke_width=0)
        top_bg.set_fill("#101520", opacity=0.55).move_to(UP * (SURF_Y + 2.0))
        bot_bg = Rectangle(width=14, height=3.6, stroke_width=0)
        bot_bg.set_fill("#0E2035", opacity=0.55).move_to(DOWN * (3.6 / 2 - SURF_Y))

        vac_lbl  = Text("Luft  (n₁ ≈ 1.0)", font=MONO, font_size=19, color=SOFT).to_corner(UL, buff=0.45).shift(DOWN * 0.5)
        med_lbl  = Text("Wasser (n₂ ≈ 1.33)", font=MONO, font_size=19, color=PRIMARY).to_corner(DL, buff=0.45).shift(UP * 0.5)

        # Incident ray
        inc_angle = 42 * DEGREES
        inc_dir   = np.array([np.sin(inc_angle), -np.cos(inc_angle), 0])
        hit_pt    = np.array([0.0, SURF_Y, 0])
        inc_start = hit_pt - inc_dir * 3.6

        # Refracted ray (Snell)
        n1, n2 = 1.0, 1.33
        sin_t = n1 * np.sin(inc_angle) / n2
        refr_angle = np.arcsin(sin_t)
        refr_dir = np.array([np.sin(refr_angle), -np.cos(refr_angle), 0])
        refr_end  = hit_pt + refr_dir * 3.2

        # Normal
        normal_line = DashedLine(hit_pt + UP * 2.4, hit_pt + DOWN * 2.4,
                                 dash_length=0.10, color=SOFT, stroke_width=1.5, stroke_opacity=0.55)

        inc_ray  = Arrow(inc_start, hit_pt, color=TEXT, buff=0, stroke_width=3.8,
                         max_tip_length_to_length_ratio=0.08)
        refr_ray = Arrow(hit_pt, refr_end, color=PRIMARY, buff=0, stroke_width=3.8,
                         max_tip_length_to_length_ratio=0.08)

        # Angles
        theta1_arc = Arc(radius=0.52, start_angle=90*DEGREES,
                         angle=inc_angle, color=TEXT, stroke_width=2.5, arc_center=hit_pt)
        theta2_arc = Arc(radius=0.52, start_angle=270*DEGREES,
                         angle=-refr_angle, color=PRIMARY, stroke_width=2.5, arc_center=hit_pt)
        theta1_lbl = MathTex(r"\theta_1", color=TEXT, font_size=24).move_to(
            hit_pt + UP * 0.78 + LEFT * 0.55)
        theta2_lbl = MathTex(r"\theta_2", color=PRIMARY, font_size=24).move_to(
            hit_pt + DOWN * 0.72 + LEFT * 0.42)

        # Huygens secondary wavelets at surface
        n_huygens = 7
        huygens_xs = np.linspace(-2.8, 2.8, n_huygens)

        def huygens_r(idx, time):
            # each wavelet grows at c/n2
            t_offset = idx / n_huygens * 1.4   # staggered hits
            return max(0, time * 0.55 - t_offset * 0.55)

        huygens_circles = always_redraw(lambda: VGroup(*[
            Circle(
                radius=max(0.01, huygens_r(i, self.renderer.time % 3.5)),
                color=SECONDARY,
                stroke_opacity=max(0, 0.5 - huygens_r(i, self.renderer.time % 3.5) * 0.25),
                stroke_width=1.8
            ).move_to([huygens_xs[i], SURF_Y, 0])
            for i in range(n_huygens)
        ]))

        # Snell formula card
        snell = MathTex(r"n_1 \sin\theta_1 = n_2 \sin\theta_2", color=ACCENT, font_size=36)
        snell_box = RoundedRectangle(corner_radius=0.13, width=snell.width + 0.55,
                                     height=snell.height + 0.36, color=ACCENT)
        snell_box.set_fill(PANEL, opacity=0.92).set_stroke(ACCENT, width=1.8, opacity=0.75)
        snell_box.move_to(snell)
        snell_card = VGroup(snell_box, snell).to_corner(UR, buff=0.52).shift(DOWN * 0.85)

        cap = caption("Wellenfronten werden gebogen, weil eine Seite früher verlangsamt wird.\n→ Snellsches Gesetz: n₁·sin θ₁ = n₂·sin θ₂")

        self.play(FadeIn(top_bg), FadeIn(bot_bg), run_time=0.5)
        self.play(Create(surface), FadeIn(vac_lbl), FadeIn(med_lbl), run_time=0.6)
        self.play(Create(normal_line), run_time=0.4)
        self.play(GrowArrow(inc_ray), run_time=0.6)
        self.play(FadeIn(huygens_circles), run_time=0.5)
        self.wait(1.5)
        self.play(GrowArrow(refr_ray),
                  Create(theta1_arc), Create(theta2_arc),
                  FadeIn(theta1_lbl), FadeIn(theta2_lbl),
                  run_time=0.9)
        self.play(FadeIn(snell_card), run_time=0.6)
        self.play(FadeIn(cap), run_time=0.4)
        self.wait(4.0)

        # ── Final summary flash ──
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.5)
        self.wait(0.15)

        summary_lines = [
            ("1. Licht = schwingendes E-Feld", TEXT),
            ("2. E-Feld regt Elektronen an (mit Phasenlag)", SOFT),
            ("3. Sekundärwellen überlagern → scheinbare Verlangsamung", PRIMARY),
            ("4. Verlangsamung = Brechungsindex n = c/v", ACCENT),
            ("5. Wellenfront-Biegung an Grenzfläche → Snell", SECONDARY),
        ]
        items = VGroup(*[
            Text(txt, font=MONO, font_size=23, color=col)
            for txt, col in summary_lines
        ]).arrange(DOWN, aligned_edge=LEFT, buff=0.28).move_to(ORIGIN)
        if items.width > 13.0:
            items.scale_to_fit_width(13.0)

        sum_title = Text("Zusammenfassung: Warum bricht Licht?", font=MONO, font_size=28, color=ACCENT)
        sum_title.to_edge(UP, buff=0.45)

        self.play(FadeIn(sum_title), run_time=0.5)
        for item in items:
            self.play(FadeIn(item, shift=UP * 0.06), run_time=0.38)
            self.wait(0.22)
        self.wait(2.5)
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.6)