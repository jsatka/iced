use iced::isolated_layer::{Blur, DropShadow, DualKawaseBlur, GaussianBlur};
use iced::widget::{button, column, container, drop_shadow, row, rule, slider, text};
use iced::{Background, Center, Color, Element, Fill, Length, Theme, Vector, color};

pub fn main() -> iced::Result {
    iced::run(Example::update, Example::view)
}

struct Example {
    blur: f32,
    distance: f32,
    angle: f32,
    algorithm: Algorithm,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Algorithm {
    Gaussian,
    DualKawase,
}

#[derive(Debug, Clone, Copy)]
enum Message {
    BlurChanged(f32),
    DistanceChanged(f32),
    AngleChanged(f32),
    Select(Algorithm),
}

impl Default for Example {
    fn default() -> Self {
        Self {
            blur: 4.0,
            distance: 5.0,
            angle: 53.13,
            algorithm: Algorithm::Gaussian,
        }
    }
}

impl Example {
    fn update(&mut self, message: Message) {
        match message {
            Message::BlurChanged(blur) => self.blur = blur,
            Message::DistanceChanged(distance) => self.distance = distance,
            Message::AngleChanged(angle) => self.angle = angle,
            Message::Select(algorithm) => self.algorithm = algorithm,
        }
    }

    fn view(&self) -> Element<'_, Message> {
        let first_paragraph =
            text("A quiet drop shadow belongs to an element, or group of elements as a whole.")
                .size(24)
                .width(Length::Fill)
                .color(Color::WHITE);

        let second_paragraph = text(
            "The two paragraphs are rendered normally onto a transparent offscreen texture for the isolated layer. When the radius is nonzero, the selected algorithm filters that captured texture. The shadow is derived from the filtered alpha at the chosen offset and tinted with the shadow color, while the original captured pixels are placed over it in the effect output. The completed layer is then composited onto the parent surface.",
        )
        .size(24)
        .width(Length::Fill)
        .color(Color::WHITE);

        let horizontal_separator = {
            let style_fn = |_theme: &Theme| rule::Style {
                color: Color::WHITE,
                radius: 0.0.into(),
                fill_mode: rule::FillMode::Padded(12),
                snap: true,
            };

            rule::horizontal(1).style(style_fn)
        };

        let vertical_content = column!(first_paragraph, horizontal_separator, second_paragraph)
            .width(560)
            .spacing(10);

        let angle = self.angle.to_radians();
        let shadow = DropShadow {
            color: Color::from_rgba(0.05, 0.02, 0.2, 0.75),
            offset: Vector::new(self.distance * angle.cos(), self.distance * angle.sin()),
            blur: match self.algorithm {
                Algorithm::Gaussian => {
                    Blur::Gaussian(GaussianBlur::new((3.0 * self.blur).min(128.0)))
                }
                Algorithm::DualKawase => {
                    Blur::DualKawase(DualKawaseBlur::new(self.blur).pyramid_depth(3))
                }
            },
        };
        let select = |label, algorithm| {
            button(label).on_press(Message::Select(algorithm)).style(
                if self.algorithm == algorithm {
                    button::primary
                } else {
                    button::secondary
                },
            )
        };

        container(
            column![
                text("Drop shadow").size(36).color(Color::WHITE),
                drop_shadow(vertical_content, shadow),
                column![
                    text(format!("Shadow distance: {:.1} px", self.distance)).color(Color::WHITE),
                    slider(0.0..=16.0, self.distance, Message::DistanceChanged,).step(0.1),
                    text(format!("Shadow angle: {:.0}°", self.angle)).color(Color::WHITE),
                    slider(0.0..=360.0, self.angle, Message::AngleChanged).step(1.0),
                    text(match self.algorithm {
                        Algorithm::Gaussian => {
                            format!("Gaussian radius: {:.1}", 3.0 * self.blur)
                        }
                        Algorithm::DualKawase => {
                            format!("Dual Kawase radius: {:.1} · depth: 3", self.blur)
                        }
                    })
                    .color(Color::WHITE),
                    slider(0.0..=16.0, self.blur, Message::BlurChanged).step(0.1),
                    row![
                        select("Gaussian", Algorithm::Gaussian),
                        select("Dual Kawase", Algorithm::DualKawase),
                    ]
                    .spacing(8),
                ]
                .spacing(8)
                .align_x(Center)
                .width(360),
            ]
            .spacing(22)
            .align_x(Center),
        )
        .center(Fill)
        .padding(48)
        .style(background)
        .into()
    }
}

fn background(_theme: &Theme) -> container::Style {
    container::Style {
        background: Some(Background::Color(color!(0x5946a7))),
        ..container::Style::default()
    }
}
