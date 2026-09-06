use iced::isolated_layer::{DualKawaseBlur, GaussianBlur};
use iced::time::Instant;
use iced::widget::{blur_dual_kawase, blur_gaussian, button, column, container, row, text};
use iced::window;
use iced::{Background, Border, Center, Color, Element, Fill, Subscription, Theme, color};

const DEFAULT_GAUSSIAN_RADIUS: f32 = 30.0;
// Gaussian radius is three times its target sigma; Kawase radius approximates
// per-axis spread. These presets are a visual starting point, not equivalence.
const DEFAULT_DUAL_KAWASE_RADIUS: f32 = 10.0;
const DUAL_KAWASE_DEPTH: u32 = 3;

pub fn main() -> iced::Result {
    iced::application(Spoiler::default, Spoiler::update, Spoiler::view)
        .subscription(Spoiler::subscription)
        .run()
}

struct Spoiler {
    algorithm: Algorithm,
    revealed: bool,
    strength: f32,
    last_frame: Instant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Algorithm {
    Gaussian,
    DualKawase,
}

#[derive(Debug, Clone, Copy)]
enum Message {
    Select(Algorithm),
    Toggle,
    Frame(Instant),
}

impl Default for Spoiler {
    fn default() -> Self {
        Self {
            algorithm: Algorithm::Gaussian,
            revealed: false,
            strength: 1.0,
            last_frame: Instant::now(),
        }
    }
}

impl Spoiler {
    fn update(&mut self, message: Message) {
        match message {
            Message::Select(algorithm) => self.algorithm = algorithm,
            Message::Toggle => self.revealed = !self.revealed,
            Message::Frame(now) => {
                let elapsed = (now - self.last_frame).as_secs_f32();
                self.last_frame = now;
                let target = if self.revealed { 0.0 } else { 1.0 };
                let step = 2.8 * elapsed;

                if self.strength < target {
                    self.strength = (self.strength + step).min(target);
                } else {
                    self.strength = (self.strength - step).max(target);
                }
            }
        }
    }

    fn view(&self) -> Element<'_, Message> {
        let spoiler = container(text("The answer is 42.").size(28).color(color!(0x18203a)))
            .padding(28)
            .style(spoiler_card);
        let blur = match self.algorithm {
            Algorithm::Gaussian => blur_gaussian(
                spoiler,
                GaussianBlur::new(DEFAULT_GAUSSIAN_RADIUS * self.strength),
            ),
            Algorithm::DualKawase => blur_dual_kawase(
                spoiler,
                DualKawaseBlur::new(DEFAULT_DUAL_KAWASE_RADIUS * self.strength)
                    .pyramid_depth(DUAL_KAWASE_DEPTH),
            ),
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
                text("Animated blur").size(36),
                row![
                    select("Gaussian", Algorithm::Gaussian),
                    select("Dual Kawase", Algorithm::DualKawase),
                ]
                .spacing(8),
                text("What is the answer?"),
                blur,
                text(match self.algorithm {
                    Algorithm::Gaussian => format!(
                        "Gaussian radius: {:.1}",
                        DEFAULT_GAUSSIAN_RADIUS * self.strength
                    ),
                    Algorithm::DualKawase => format!(
                        "Dual Kawase radius: {:.1} · depth: {DUAL_KAWASE_DEPTH}",
                        DEFAULT_DUAL_KAWASE_RADIUS * self.strength
                    ),
                }),
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
