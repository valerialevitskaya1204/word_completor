import reflex as rx

config = rx.Config(
    app_name="word_completor_ui",
    plugins=[
        rx.plugins.SitemapPlugin(),
        rx.plugins.TailwindV4Plugin(),
    ]
)