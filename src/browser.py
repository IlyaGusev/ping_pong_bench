import json
import sys
import shutil
from typing import Dict, Any, Type, Optional, cast

from textual import on, events, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import (
    Header,
    Footer,
    MarkdownViewer,
    Static,
    Input,
    Label,
    RadioButton,
    RadioSet,
    Button,
)
from textual.widget import Widget
from textual.validation import Number
from textual.containers import Container, Grid, Vertical
from textual.screen import ModalScreen


def to_markdown(record: Dict[str, Any]) -> str:
    result = ""
    messages = record["messages"]
    for m in messages:
        content = m["content"]
        content = content.replace("*", "**")
        result += "# {role}\n{content}\n\n".format(role=m["role"], content=content)
    return result


def to_meta(record: Dict[str, Any]) -> str:
    meta = {k: v for k, v in record.items() if k != "messages"}
    result = []
    for k, v in meta.items():
        if k == "character":
            result.append(v["char_name"])
        elif k == "human_scores":
            result.append(str(v))
    return ", ".join(result)


class RateScreen(ModalScreen[bool]):
    BINDINGS = [
        ("q", "app.pop_screen", "Pop screen"),
        ("1", "select(1)", ""),
        ("2", "select(2)", ""),
        ("3", "select(3)", ""),
        ("4", "select(4)", ""),
        ("5", "select(5)", ""),
    ]

    def compose(self) -> ComposeResult:
        with Vertical(id="rate_dialog"):
            yield Label("Bot Interaction Rating", id="title")
            questions = (
                (
                    "in_character",
                    "The bot's answers are perfectly aligned with an assigned character",
                ),
                (
                    "entertaining",
                    "The bot's responses are extremely engaging and entertaining",
                ),
                (
                    "fluency",
                    "The bot's language use is of the highest quality, without any mistakes",
                ),
            )
            for set_id, question in questions:
                yield Label(question)
                yield RadioSet(
                    RadioButton("1. Strongly disagree"),
                    RadioButton("2. Disagree"),
                    RadioButton("3. Neutral"),
                    RadioButton("4. Agree"),
                    RadioButton("5. Strongly agree"),
                    id=set_id,
                )
            with Vertical(id="button_row"):
                yield Button("Submit", variant="primary", id="submit")

    def action_select(self, number: int) -> None:
        current_radioset = self.get_current_focus()
        if isinstance(current_radioset, RadioSet):
            button = current_radioset.children[number - 1]
            assert isinstance(button, RadioButton)
            button.toggle()
            self.focus_next()

    def get_current_focus(self) -> Optional[Widget]:
        for widget_id in ["in_character", "entertaining", "fluency", "submit"]:
            widget = self.query_one(f"#{widget_id}")
            if widget.has_focus:
                return widget
        return None

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "submit":
            self.dismiss(True)

    def get_ratings(self) -> Dict[str, int]:
        ids = ("in_character", "entertaining", "fluency")
        labels = []
        for i in ids:
            radio_set = cast(RadioSet, self.query_one("#" + i))
            button = radio_set.pressed_button
            assert button is not None
            label = int(str(button.label)[0])
            labels.append(label)
        return {i: l for i, l in zip(ids, labels)}

    def is_visible(self) -> bool:
        return self.app.screen is self


class Browser(App[None]):
    CSS_PATH = "browser.tcss"
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("d", "delete", "Delete"),
        ("b", "back", "Back"),
        ("f", "forward", "Forward"),
        ("s", "save", "Save"),
        ("r", "rate", "Rate"),
        Binding("g", "go", "Go", show=False, priority=True),
    ]

    def compose(self) -> ComposeResult:
        self.path = sys.argv[1]
        self.current_idx = 0
        with open(sys.argv[1]) as r:
            self.records = [json.loads(line) for line in r]
        yield Header()
        yield Static("", id="meta")
        yield Container(MarkdownViewer(), Static("Loading...", id="loading"), id="main-content")
        yield Static("", id="counter")
        yield Input(
            placeholder="Enter index", validators=[Number()], restrict="[0-9]*", valid_empty=True
        )
        yield Footer()

    @property
    def markdown_viewer(self) -> MarkdownViewer:
        return self.query_one(MarkdownViewer)

    @property
    def footer(self) -> Footer:
        return self.query_one(Footer)

    @property
    def header(self) -> Header:
        return self.query_one(Header)

    @property
    def meta_info(self) -> Static:
        return cast(Static, self.query_one("#meta"))

    @property
    def counter(self) -> Static:
        return cast(Static, self.query_one("#counter"))

    @property
    def input(self) -> Input:
        return self.query_one(Input)

    @property
    def loading_indicator(self) -> Static:
        return cast(Static, self.query_one("#loading"))

    async def show_record(self) -> None:
        if len(self.records) == 0:
            await self.markdown_viewer.document.update("No records left")
            self.counter.update("No records")
            return

        assert self.current_idx < len(self.records)

        self.markdown_viewer.display = False
        self.loading_indicator.display = True

        record = self.records[self.current_idx]
        self.meta_info.update(to_meta(record))
        await self.markdown_viewer.document.update(to_markdown(record))
        self.counter.update(f"Record {self.current_idx + 1} of {len(self.records)}")

        def show_markdown() -> None:
            self.markdown_viewer.focus()
            self.markdown_viewer.display = True
            self.loading_indicator.display = False

        self.markdown_viewer.scroll_home(animate=False, on_complete=show_markdown)

    async def on_mount(self) -> None:
        self.loading_indicator.display = False
        await self.show_record()

    @on(Input.Submitted)
    async def goto(self, event: Input.Submitted) -> None:
        if not event.validation_result or not event.validation_result.is_valid:
            self.notify(
                "Invalid index. Please enter a number between 1 and {}.".format(len(self.records))
            )
            return

        input_value = self.input.value
        index = int(input_value) - 1
        if 0 <= index < len(self.records):
            self.current_idx = index
            await self.show_record()
        else:
            self.notify(
                "Invalid index. Please enter a number between 1 and {}.".format(len(self.records))
            )
        self.input.clear()

    async def action_back(self) -> None:
        self.current_idx -= 1
        if self.current_idx < 0:
            self.current_idx = len(self.records) - 1
        await self.show_record()

    async def action_forward(self) -> None:
        self.current_idx += 1
        if self.current_idx >= len(self.records):
            self.current_idx = 0
        await self.show_record()

    async def action_delete(self) -> None:
        assert 0 <= self.current_idx < len(self.records)
        self.records.pop(self.current_idx)
        if self.current_idx >= len(self.records):
            self.current_idx = 0
        await self.show_record()

    async def action_go(self) -> None:
        if self.input.has_focus:
            validation_result = self.input.validate(self.input.value)
            self.post_message(self.input.Submitted(self.input, self.input.value, validation_result))

    def is_rate_screen_active(self) -> bool:
        return any(isinstance(screen, RateScreen) for screen in self.screen_stack)

    def on_key(self, event: events.Key) -> None:
        if (
            not self.is_rate_screen_active()
            and event.key in "1234567890"
            and not self.input.has_focus
        ):
            self.input.focus()
            self.input.value = event.key

    def action_save(self) -> None:
        with open(self.path + "_tmp", "w") as w:
            for record in self.records:
                w.write(json.dumps(record, ensure_ascii=False) + "\n")
        shutil.move(self.path + "_tmp", self.path)
        self.notify("Saved!")

    @work
    async def action_rate(self) -> None:
        screen = RateScreen()
        is_ok = await self.push_screen_wait(screen)
        if is_ok:
            ratings = screen.get_ratings()
            record = self.records[self.current_idx]
            record["human_scores"] = ratings
            self.meta_info.update(to_meta(record))
            self.notify("Scores updated!")


if __name__ == "__main__":
    Browser().run()
