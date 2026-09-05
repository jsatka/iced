use iced::widget::{column, container, opacity, row, slider, text};
use iced::{Background, Border, Center, Color, Element, Fill, Theme, Vector, color};

pub fn main() -> iced::Result {
    iced::run(GroupOpacity::update, GroupOpacity::view)
}

struct GroupOpacity {
    opacity: f32,
}

#[derive(Debug, Clone, Copy)]
enum Message {
    OpacityChanged(f32),
}

impl Default for GroupOpacity {
    fn default() -> Self {
        Self { opacity: 0.45 }
    }
}

impl GroupOpacity {
    fn update(&mut self, message: Message) {
        match message {
            Message::OpacityChanged(opacity) => self.opacity = opacity,
        }
    }

    fn view(&self) -> Element<'_, Message> {
        let title = text("Group opacity").size(36);
        let subtitle =
            text("The entire card is captured once in isolation, then composited uniformly.");

        let opacity_controls = column![
            text(format!("Opacity: {:.0}%", self.opacity * 100.0)),
            slider(0.0..=1.0, self.opacity, Message::OpacityChanged).step(0.01),
        ]
        .spacing(8)
        .width(360);

        let swatches_row = row![
            swatch('A', color!(0xff5c7a)),
            swatch('B', color!(0x52d6c7)),
            swatch('C', color!(0x6d7cff)),
        ]
        .spacing(24);

        let captured_group = container(
            column![
                text("Text, backgrounds, and nested shadow effect in this container are first rendered onto a layer backed by an offscreen GPU texture.")
                    .wrapping(text::Wrapping::Word),
                swatches_row,
            ]
            .align_x(Center)
            .width(400)
            .spacing(12),
        )
        .padding(24)
        .style(card);

        container(
            column![
                title,
                subtitle,
                opacity_controls,
                opacity(captured_group, self.opacity),
            ]
            .spacing(24)
            .align_x(Center),
        )
        .center(Fill)
        .style(background)
        .into()
    }
}

fn swatch(letter: char, color: Color) -> iced::widget::Container<'static, Message> {
    use iced::widget::{drop_shadow, isolated_layer::DropShadow};

    let shadow = DropShadow {
        color: Color::from_rgba(0.05, 0.02, 0.2, 0.75),
        offset: Vector::new(2.0, 2.0),
        blur_radius: 0.0,
    };

    let content = text(letter).color(Color::WHITE).size(48);

    container(drop_shadow(content, shadow))
        .center(90)
        .style(move |_| container::Style {
            background: Some(Background::Color(color)),
            border: Border::default().rounded(18),
            ..container::Style::default()
        })
}

fn card(_theme: &Theme) -> container::Style {
    container::Style {
        background: Some(Background::Color(color!(0xffffff))),
        text_color: Some(color!(0x16213d)),
        border: Border::default().rounded(24),
        ..container::Style::default()
    }
}

fn background(_theme: &Theme) -> container::Style {
    container::Style {
        background: Some(Background::Color(color!(0x17213d))),
        text_color: Some(Color::WHITE),
        ..container::Style::default()
    }
}
