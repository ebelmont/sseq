/// Catppuccin Mocha palette for diagram rendering.
///
/// Colors are provided as tiny-skia `Color` values via functions
/// (since `Color::from_rgba8` is not const).
pub mod mocha {
    use tiny_skia::Color;

    // Base layers
    pub fn crust() -> Color { Color::from_rgba8(0x11, 0x11, 0x1b, 255) }
    pub fn mantle() -> Color { Color::from_rgba8(0x18, 0x18, 0x25, 255) }
    pub fn base() -> Color { Color::from_rgba8(0x1e, 0x1e, 0x2e, 255) }
    pub fn surface0() -> Color { Color::from_rgba8(0x31, 0x32, 0x44, 255) }
    pub fn surface1() -> Color { Color::from_rgba8(0x45, 0x47, 0x5a, 255) }
    pub fn surface2() -> Color { Color::from_rgba8(0x58, 0x5b, 0x70, 255) }
    pub fn overlay0() -> Color { Color::from_rgba8(0x6c, 0x70, 0x86, 255) }
    pub fn overlay1() -> Color { Color::from_rgba8(0x7f, 0x84, 0x9c, 255) }

    // Text
    pub fn subtext0() -> Color { Color::from_rgba8(0xa6, 0xad, 0xc8, 255) }
    pub fn subtext1() -> Color { Color::from_rgba8(0xba, 0xc2, 0xde, 255) }
    pub fn text() -> Color { Color::from_rgba8(0xcd, 0xd6, 0xf4, 255) }

    // Accents
    pub fn lavender() -> Color { Color::from_rgba8(0xb4, 0xbe, 0xfe, 255) }
    pub fn blue() -> Color { Color::from_rgba8(0x89, 0xb4, 0xfa, 255) }
    pub fn sapphire() -> Color { Color::from_rgba8(0x74, 0xc7, 0xec, 255) }
    pub fn sky() -> Color { Color::from_rgba8(0x89, 0xdc, 0xeb, 255) }
    pub fn teal() -> Color { Color::from_rgba8(0x94, 0xe2, 0xd5, 255) }
    pub fn green() -> Color { Color::from_rgba8(0xa6, 0xe3, 0xa1, 255) }
    pub fn yellow() -> Color { Color::from_rgba8(0xf9, 0xe2, 0xaf, 255) }
    pub fn peach() -> Color { Color::from_rgba8(0xfa, 0xb3, 0x87, 255) }
    pub fn maroon() -> Color { Color::from_rgba8(0xeb, 0xa0, 0xac, 255) }
    pub fn red() -> Color { Color::from_rgba8(0xf3, 0x8b, 0xa8, 255) }
    pub fn mauve() -> Color { Color::from_rgba8(0xcb, 0xa6, 0xf7, 255) }
    pub fn pink() -> Color { Color::from_rgba8(0xf5, 0xc2, 0xe7, 255) }

    /// Transparent background for dark terminal compositing.
    pub fn transparent() -> Color { Color::from_rgba8(0, 0, 0, 0) }
}
