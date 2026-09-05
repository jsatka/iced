use iced::isolated_layer::AlphaMask;
use iced::widget::{alpha_mask, column, container, scrollable, text};
use iced::{Background, Border, Center, Color, Element, Fill, Theme, color};

pub fn main() -> iced::Result {
    iced::run(Example::update, Example::view)
}

#[derive(Default)]
struct Example;

impl Example {
    fn update(&mut self, _message: ()) {}

    fn view(&self) -> Element<'_, ()> {
        let items = (1..=30).fold(column![].spacing(8).padding([34, 18]), |items, index| {
            items.push(
                container(text(format!("Scrollable row {index:02}")))
                    .width(Fill)
                    .padding(14)
                    .style(move |_theme| row_style(index)),
            )
        });

        let scrolling = container(scrollable(items).height(420))
            .width(460)
            .height(420)
            .style(frame);

        container(
            column![
                text("Gradient alpha mask").size(36),
                text("Scroll: content fades through generated top and bottom gradients."),
                alpha_mask(scrolling, AlphaMask::vertical(36.0, 52.0)),
            ]
            .spacing(20)
            .align_x(Center),
        )
        .center(Fill)
        .style(background)
        .into()
    }
}

fn row_style(index: usize) -> container::Style {
    let color = if index.is_multiple_of(2) {
        color!(0x273b68)
    } else {
        color!(0x304a82)
    };

    container::Style {
        background: Some(Background::Color(color)),
        border: Border::default().rounded(12),
        ..container::Style::default()
    }
}

fn frame(_theme: &Theme) -> container::Style {
    container::Style {
        background: Some(Background::Color(color!(0x172442))),
        border: Border::default().rounded(22),
        ..container::Style::default()
    }
}

fn background(_theme: &Theme) -> container::Style {
    container::Style {
        background: Some(Background::Color(color!(0x0c1428))),
        text_color: Some(Color::WHITE),
        ..container::Style::default()
    }
}
