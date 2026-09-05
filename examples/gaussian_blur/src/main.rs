use iced::time::Instant;
use iced::widget::{blur, button, column, container, text};
use iced::window;
use iced::{Background, Border, Center, Color, Element, Fill, Subscription, Theme, color};

pub fn main() -> iced::Result {
    iced::application(Spoiler::default, Spoiler::update, Spoiler::view)
        .subscription(Spoiler::subscription)
        .run()
}

struct Spoiler {
    revealed: bool,
    sigma: f32,
    last_frame: Instant,
}

#[derive(Debug, Clone, Copy)]
enum Message {
    Toggle,
    Frame(Instant),
}

impl Default for Spoiler {
    fn default() -> Self {
        Self {
            revealed: false,
            sigma: 10.0,
            last_frame: Instant::now(),
        }
    }
}

impl Spoiler {
    fn update(&mut self, message: Message) {
        match message {
            Message::Toggle => self.revealed = !self.revealed,
            Message::Frame(now) => {
                let elapsed = (now - self.last_frame).as_secs_f32();
                self.last_frame = now;
                let target = if self.revealed { 0.0 } else { 10.0 };
                let step = 28.0 * elapsed;

                if self.sigma < target {
                    self.sigma = (self.sigma + step).min(target);
                } else {
                    self.sigma = (self.sigma - step).max(target);
                }
            }
        }
    }

    fn view(&self) -> Element<'_, Message> {
        let spoiler = container(text("The answer is 42.").size(28).color(color!(0x18203a)))
            .padding(28)
            .style(spoiler_card);

        container(
            column![
                text("Animated Gaussian blur").size(36),
                text("What is the answer?"),
                blur(spoiler, self.sigma),
                button(if self.revealed {
                    "Hide spoiler"
                } else {
                    "Reveal spoiler"
                })
                .on_press(Message::Toggle),
            ]
            .spacing(24)
            .align_x(Center),
        )
        .center(Fill)
        .style(background)
        .into()
    }

    fn subscription(&self) -> Subscription<Message> {
        window::frames().map(Message::Frame)
    }
}

fn spoiler_card(_theme: &Theme) -> container::Style {
    container::Style {
        background: Some(Background::Color(color!(0xf8d56b))),
        border: Border::default().rounded(18),
        ..container::Style::default()
    }
}

fn background(_theme: &Theme) -> container::Style {
    container::Style {
        background: Some(Background::Color(color!(0x11182d))),
        text_color: Some(Color::WHITE),
        ..container::Style::default()
    }
}
