import reflex as rx

import nltk

import pandas as pd
import ast
from pathlib import Path

from models import NGramLanguageModel, WordCompletor, TextSuggestion
from logger import setup_logger

logger = setup_logger(__name__)

nltk.download("punkt")


def _find_dataset_path() -> str:
    candidates = [
        Path(r"C:\Users\valer\.vscode\word_completor\data\cleaned_emals.csv"),
        Path(__file__).resolve().parent / "data" / "cleaned_emals.csv",
        Path(__file__).resolve().parents[1] / "data" / "cleaned_emals.csv",
    ]
    for p in candidates:
        if p.exists():
            logger.info("Resolved dataset path: %s", p)
            return str(p)
    logger.warning("Dataset not found in known locations.")
    return ""

DATASET_PATH = _find_dataset_path()


def _parse_tokenized_series(series: pd.Series) -> list[list[str]]:
    corpus: list[list[str]] = []
    for s in series.dropna().tolist():
        try:
            lst = ast.literal_eval(s)
            if isinstance(lst, list):
                corpus.append([str(x) for x in lst])
        except Exception as e:
            logger.warning("Failed to parse row: %s", e)
    return corpus


class AppState(rx.State):
    csv_path: str = DATASET_PATH
    status: str = (
        "Dataset path resolved. Set limits (optional) and click 'Load CSV'."
        if DATASET_PATH else
        "Dataset not found. Set the correct path and click 'Load CSV'."
    )

    input_text: str = ""
    n: int = 2
    n_words: int = 3
    n_texts: int = 1
    row_limit: int = 100000  #комп умирал от 500к наблюдений

    completions: list[str] = []
    prefix_candidates: list[tuple[str, float]] = []

    _corpus: list[list[str]] = []
    _word_completor: WordCompletor | None = None
    _ngram: NGramLanguageModel | None = None
    _sugg: TextSuggestion | None = None

    def set_csv_path(self, v: str):
        logger.info("CSV path updated to %s", v)
        self.csv_path = v

    def set_row_limit(self, v: str):
        try:
            self.row_limit = max(1, int(v))
            logger.info("Set row_limit=%d", self.row_limit)
        except Exception:
            logger.warning("Invalid row_limit: %r", v)

    def set_n(self, v: str):
        try:
            self.n = max(1, int(v))
            logger.info("Set n=%d", self.n)
        except Exception:
            return
        if self._corpus:
            self._build_models()

    def set_n_words(self, v: str):
        try:
            self.n_words = max(0, int(v))
            logger.info("Set n_words=%d", self.n_words)
        except Exception:
            return

    def set_n_texts(self, v: str):
        try:
            self.n_texts = max(1, int(v))
            logger.info("Set n_texts=%d", self.n_texts)
        except Exception:
            return

    def load_csv(self):
        try:
            logger.info("Loading CSV: %s", self.csv_path)
            path = (self.csv_path or "").strip()
            if not path:
                self.status = "Это не путь к датасету"
                logger.warning(self.status)
                return
            if not Path(path).exists():
                self.status = f"Не найден файл с датасетом: {path}"
                logger.error(self.status)
                return

            df = pd.read_csv(path, nrows=self.row_limit)
            logger.info("Read CSV with %d rows (capped), columns=%s", len(df), df.columns.tolist())

            corpus = _parse_tokenized_series(df["tokenized_message"])

            self._corpus = corpus
            self._build_models()
            self.status = f"Loaded {len(corpus)} rows from: {path} (limit={self.row_limit})"
            logger.info(self.status)
        except Exception:
            logger.exception("Error loading CSV")
            self.status = "Error loading CSV (see logs)."

    def _build_models(self):
        try:
            logger.info("Building models...")
            self._word_completor = WordCompletor(self._corpus)
            self._ngram = NGramLanguageModel(corpus=self._corpus, n=self.n)
            self._sugg = TextSuggestion(self._word_completor, self._ngram)
            logger.info("Models built successfully")
        except Exception:
            logger.exception("Error building models")
            self.status = "Error building models."

    def on_text_change(self, value: str):
        self.input_text = value
        logger.debug("Text changed: %s", value)
        if not self._word_completor:
            self.prefix_candidates = []
            return
        last = value.strip().split()[-1] if value.strip() else ""
        if not last:
            self.prefix_candidates = []
            return
        words, probs = self._word_completor.get_words_and_probs(last)
        self.prefix_candidates = list(zip(words[:10], probs[:10]))

    def suggest(self):
        if not self._sugg:
            self.status = "Load a corpus first."
            logger.warning(self.status)
            return
        text = self.input_text.strip().split() if self.input_text.strip() else []
        raw = self._sugg.suggest_text(text, n_words=self.n_words, n_texts=self.n_texts)
        self.completions = [" ".join(s) for s in raw if isinstance(s, list)]
        logger.info("Completions generated: %s", self.completions)

    def accept_first(self):
        if not self.completions:
            return
        first = self.completions[0]
        if not first:
            return
        self.input_text = (self.input_text + " " + first).strip()
        logger.info("Accepted suggestion: %s", first)
        self.on_text_change(self.input_text)

    @rx.var
    def prefix_lines(self) -> str:
        try:
            return "\n".join(f"{w} · {p:.4f}" for (w, p) in self.prefix_candidates)
        except Exception:
            return ""

    @rx.var
    def has_completions(self) -> bool:
        try:
            return len(self.completions) > 0
        except Exception:
            return False

def stat_badge(label: str, value: str) -> rx.Component:
    return rx.hstack(
        rx.text(label, size="2", color_scheme="gray"),
        rx.spacer(),
        rx.code(value, size="2"),
        align="center",
        width="100%",
        padding="8px",
        border="1px solid",
        border_color="gray.3",
        radius="lg",
        bg="gray.1",
    )

def header() -> rx.Component:
    return rx.vstack(
        rx.text("Word Completor + N-gram", size="7", weight="bold"),
        rx.text("Reflex UI", color="gray.9"),
        spacing="2",
        align_items="start",
    )

def control_panel() -> rx.Component:
    return rx.vstack(
        rx.hstack(
            rx.input(
                value=AppState.csv_path,
                placeholder="Путь к датасету",
                on_change=AppState.set_csv_path,
                width="100%",
            ),
            rx.button("Подгрузить CSV", on_click=AppState.load_csv),
            spacing="3",
            width="100%",
        ),
        rx.hstack(
            stat_badge("Статус", AppState.status),
        ),
        rx.hstack(
            rx.vstack(
                rx.text("Лимит строк (~100k)"),
                rx.input(
                    type="number",
                    value=AppState.row_limit,
                    on_change=AppState.set_row_limit,
                    min=1,
                    step=1000,
                    width="100%",
                ),
                width="25%",
            ),
            rx.vstack(
                rx.text("n (размер контекста)"),
                rx.input(
                    type="number",
                    value=AppState.n,
                    on_change=AppState.set_n,
                    min=1,
                    max=5,
                    step=1,
                    width="100%",
                ),
                width="25%",
            ),
            rx.vstack(
                rx.text("n_words"),
                rx.input(
                    type="number",
                    value=AppState.n_words,
                    on_change=AppState.set_n_words,
                    min=0,
                    max=10,
                    step=1,
                    width="100%",
                ),
                width="25%",
            ),
            rx.vstack(
                rx.text("сколько рекомендовать"),
                rx.input(
                    type="number",
                    value=AppState.n_texts,
                    on_change=AppState.set_n_texts,
                    min=1,
                    max=5,
                    step=1,
                    width="100%",
                ),
                width="25%",
            ),
            spacing="3",
            width="100%",
        ),
        spacing="4",
        width="100%",
    )

def input_area() -> rx.Component:
    return rx.vstack(
        rx.text("Введите текст:", weight="medium"),
        rx.text_area(
            value=AppState.input_text,
            on_change=AppState.on_text_change,
            placeholder="Начинайте печатать…",
            rows="4",
            width="100%",
        ),
        rx.hstack(
            rx.button("Предложить", on_click=AppState.suggest),
            rx.button("Принять первое предложение", on_click=AppState.accept_first, color_scheme="green"),
            spacing="3",
        ),
        spacing="3",
        width="100%",
    )

def live_prefix_box() -> rx.Component:
    return rx.card(
        rx.vstack(
            rx.text("Рекомендованные префиксы (топ-10)", weight="medium"),
            rx.code(AppState.prefix_lines, wrap="wrap"),
            spacing="2",
            width="100%",
        ),
        width="100%",
        radius="xl",
        padding="16px",
    )

def suggestions_box() -> rx.Component:
    def render_row(row: str) -> rx.Component:
        return rx.badge(row, size="2", variant="surface")

    return rx.card(
        rx.vstack(
            rx.text("Автодополнения", weight="medium"),
            rx.cond(
                AppState.has_completions,
                rx.vstack(
                    rx.foreach(AppState.completions, render_row),
                    spacing="2",
                    width="100%",
                ),
                rx.text("Пока нет дополнений."),
            ),
            spacing="2",
            width="100%",
        ),
        width="100%",
        radius="xl",
        padding="16px",
    )

def index() -> rx.Component:
    return rx.container(
        rx.vstack(
            header(),
            control_panel(),
            rx.hstack(
                rx.box(input_area(), width="50%"),
                rx.box(live_prefix_box(), width="50%"),
                width="100%",
                spacing="4",
                align_items="start",
            ),
            suggestions_box(),
            spacing="6",
            padding_y="6",
        ),
        size="4",
    )

app = rx.App()
app.add_page(index, title="Автодополнение с помощью н грамной модели")
