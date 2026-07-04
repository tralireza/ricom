use config::Config;

#[test]
fn composed_spec_roundtrips_through_to_toml() {
    let src = r#"
[anim]
duration = 0.2
close = "fade"

[anim.open]
duration = 0.3
blocks = [
  { block = "opacity", easing = "ease-out" },
  { block = "translate", dy = -60.0, easing = "ease-in" },
]
"#;
    let c: Config = toml::from_str(src).unwrap();
    let printed = c.to_toml();
    let back: Config = toml::from_str(&printed).unwrap_or_else(|e| {
        panic!("re-parse of to_toml() failed: {e}\n--- printed ---\n{printed}");
    });
    assert_eq!(c, back, "roundtrip changed the config\n--- printed ---\n{printed}");
}
