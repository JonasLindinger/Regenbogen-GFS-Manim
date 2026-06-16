from manim import *
import numpy as np


class PrismRefraction(Scene):
    def construct(self):

        # --------------------------------------------------
        # Prism
        # --------------------------------------------------
        prism = Polygon(
            [-1, -2, 0],
            [3, 0, 0],
            [-1, 2, 0],
            color=WHITE,
            fill_color=BLUE,
            fill_opacity=0.15,
        )

        self.play(Create(prism))

        air_label = Text("Air", font_size=28).move_to(LEFT * 5)
        glass_label = Text("Glass (v = 0.67c)", font_size=28).move_to(RIGHT * 1.5)

        self.play(FadeIn(air_label), FadeIn(glass_label))

        # --------------------------------------------------
        # Geometry
        # --------------------------------------------------

        incident_angle = 20 * DEGREES

        # Incoming direction
        d_air = np.array([
            np.cos(incident_angle),
            np.sin(incident_angle),
            0,
        ])

        # Wavefront orientation
        n_air = np.array([
            -np.sin(incident_angle),
            np.cos(incident_angle),
            0,
        ])

        # Prism index
        n1 = 1.0
        n2 = 1.5

        # Surface normal of left prism face
        surface_angle = 60 * DEGREES

        normal = np.array([
            np.cos(surface_angle),
            np.sin(surface_angle),
            0,
        ])

        # Snell's law
        theta_i = 40 * DEGREES

        theta_t = np.arcsin(n1 / n2 * np.sin(theta_i))

        refracted_angle = -10 * DEGREES

        d_glass = np.array([
            np.cos(refracted_angle),
            np.sin(refracted_angle),
            0,
        ])

        n_glass = np.array([
            -np.sin(refracted_angle),
            np.cos(refracted_angle),
            0,
        ])

        # --------------------------------------------------
        # Wavefront creation
        # --------------------------------------------------

        wavefronts = VGroup()

        spacing = 0.5
        length = 8

        start_offset = -10

        for i in range(35):
            s = start_offset + i * spacing

            line = Line(
                ORIGIN - n_air * length / 2,
                ORIGIN + n_air * length / 2,
                color=YELLOW,
                stroke_width=4,
            )

            line.shift(d_air * s)

            wavefronts.add(line)

        self.add(wavefronts)

        prism_path = prism

        # --------------------------------------------------
        # Update logic
        # --------------------------------------------------

        c_air = 2.0
        c_glass = 1.34  # ~0.67c

        def update_wavefront(line, dt):

            if not hasattr(line, "phase"):
                line.phase = np.dot(line.get_center(), d_air)

            line.phase += c_air * dt

            center_air = d_air * line.phase

            sample_left = center_air - n_air * 2
            sample_right = center_air + n_air * 2

            left_inside = prism_path.get_vertices() is not None and prism_path.point_from_proportion(
                0
            ) is not None

            # Determine how much of the front is inside.
            # Approximate using endpoint containment.
            p1 = sample_left[:2]
            p2 = sample_right[:2]

            inside_left = prism_path.get_center()[0] - 0.2 < p1[0] < 3
            inside_right = prism_path.get_center()[0] - 0.2 < p2[0] < 3

            frac = 0

            if inside_left and inside_right:
                frac = 1
            elif inside_left or inside_right:
                frac = 0.5

            # Interpolate orientation
            direction = (
                (1 - frac) * d_air +
                frac * d_glass
            )

            direction /= np.linalg.norm(direction)

            normal_front = np.array([
                -direction[1],
                direction[0],
                0,
            ])

            # Speed interpolation
            speed = (1 - frac) * c_air + frac * c_glass

            if not hasattr(line, "glass_phase"):
                line.glass_phase = 0

            line.glass_phase += speed * dt

            center = direction * line.glass_phase + LEFT * 8

            new_line = Line(
                center - normal_front * length / 2,
                center + normal_front * length / 2,
            )

            line.become(
                new_line.set_color(YELLOW).set_stroke(width=4)
            )

        for wf in wavefronts:
            wf.add_updater(update_wavefront)

        # --------------------------------------------------
        # Explanatory text
        # --------------------------------------------------

        title = Text(
            "Wavefronts slow in glass → front rotates → light refracts",
            font_size=32,
        ).to_edge(UP)

        self.play(Write(title))

        self.wait(10)

        for wf in wavefronts:
            wf.clear_updaters()

        # --------------------------------------------------
        # Explanation overlay
        # --------------------------------------------------

        explanation = VGroup(
            Text(
                "One side enters glass first",
                font_size=28
            ),
            Text(
                "That side slows to ~0.67c",
                font_size=28
            ),
            Text(
                "The wavefront pivots",
                font_size=28
            ),
            Text(
                "The propagation direction changes",
                font_size=28
            ),
            Text(
                "Different colors bend differently → rainbow",
                font_size=28
            ),
        ).arrange(DOWN, aligned_edge=LEFT)

        explanation.to_corner(DR)

        self.play(FadeIn(explanation))
        self.wait(4)