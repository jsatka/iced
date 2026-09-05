use iced::isolated_layer::DropShadow;
use iced::widget::{column, container, drop_shadow, slider, text};
use iced::{Background, Center, Color, Element, Fill, Theme, Vector, color};

pub fn main() -> iced::Result {
    iced::run(Example::update, Example::view)
}

struct Example {
    blur: f32,
}

#[derive(Debug, Clone, Copy)]
enum Message {
    BlurChanged(f32),
}

impl Default for Example {
    fn default() -> Self {
        Self { blur: 4.0 }
    }
}

impl Example {
    fn update(&mut self, message: Message) {
        match message {
            Message::BlurChanged(blur) => self.blur = blur,
        }
    }

    fn view(&self) -> Element<'_, Message> {
        let paragraph = text(
            "A quiet drop shadow belongs to an element, or group of elements as a whole. First, the glyphs in this paragraph are first drawn with the normal paragraph rendering onto a separate offscreen texture. Then, the alpha channel is used to compute and render the drop shadow onto the isolated layer output surface before compositing the separately captured paragraph pixels on top.",
        )
        .size(30)
        .width(560)
        .color(Color::WHITE);

        let shadow = DropShadow {
            color: Color::from_rgba(0.05, 0.02, 0.2, 0.75),
            offset: Vector::new(3.0, 4.0),
            blur_radius: self.blur,
        };

        container(
            column![
                text("Drop shadow").size(36).color(Color::WHITE),
                column![
                    text(format!("Shadow blur: {:.1}", self.blur)).color(Color::WHITE),
                    slider(0.0..=4.0, self.blur, Message::BlurChanged).step(0.1),
                ]
                .spacing(8)
                .width(360),
                drop_shadow(paragraph, shadow),
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
