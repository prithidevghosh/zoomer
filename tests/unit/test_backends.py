"""Unit tests for the input backends.

The desktop backend is tested against a stand-in for pynput's controllers, so
the tests assert exactly which keystrokes and wheel events reach the operating
system without needing a desktop session or input permissions.
"""

from __future__ import annotations

import contextlib
import sys
from dataclasses import dataclass, field
from typing import Any

import pytest

from zoomer.backends import (
    BACKEND_NAMES,
    DesktopBackend,
    InputBackend,
    NoopBackend,
    RecordingBackend,
    create_backend,
    dispatch,
)
from zoomer.backends.desktop import resolve_zoom_key
from zoomer.types import ScrollEvent, ZoomEvent


class TestDispatch:
    def test_routes_a_zoom_event_to_the_zoom_method(self) -> None:
        backend = RecordingBackend()
        dispatch(backend, [ZoomEvent(2)])
        assert backend.zoom_calls == [2]
        assert backend.scroll_calls == []

    def test_routes_a_scroll_event_to_the_scroll_method(self) -> None:
        backend = RecordingBackend()
        dispatch(backend, [ScrollEvent(-3)])
        assert backend.scroll_calls == [-3]
        assert backend.zoom_calls == []

    def test_preserves_the_order_events_were_produced_in(self) -> None:
        backend = RecordingBackend()
        dispatch(backend, [ZoomEvent(1), ScrollEvent(2), ZoomEvent(-1)])
        assert backend.zoom_calls == [1, -1]
        assert backend.scroll_calls == [2]

    def test_an_empty_batch_does_nothing(self) -> None:
        backend = RecordingBackend()
        dispatch(backend, [])
        assert backend.zoom_calls == backend.scroll_calls == []

    def test_an_unrecognised_event_type_is_rejected_loudly(self) -> None:
        with pytest.raises(TypeError, match="unsupported gesture event"):
            dispatch(RecordingBackend(), ["not an event"])  # type: ignore[list-item]


class TestRecordingBackend:
    def test_satisfies_the_backend_protocol(self) -> None:
        assert isinstance(RecordingBackend(), InputBackend)

    def test_sums_net_movement_for_convenient_assertions(self) -> None:
        backend = RecordingBackend()
        dispatch(backend, [ZoomEvent(3), ZoomEvent(-1), ScrollEvent(2), ScrollEvent(2)])
        assert backend.net_zoom == 2
        assert backend.net_scroll == 4

    def test_reports_a_clean_slate_before_any_events(self) -> None:
        backend = RecordingBackend()
        assert backend.net_zoom == 0
        assert backend.net_scroll == 0
        assert backend.closed is False

    def test_records_shutdown(self) -> None:
        backend = RecordingBackend()
        backend.close()
        assert backend.closed is True

    def test_clear_forgets_history_without_disabling_the_backend(self) -> None:
        backend = RecordingBackend()
        backend.zoom(5)
        backend.clear()
        backend.zoom(2)
        assert backend.zoom_calls == [2]

    def test_instances_do_not_share_recorded_history(self) -> None:
        first, second = RecordingBackend(), RecordingBackend()
        first.zoom(1)
        assert second.zoom_calls == []


class TestNoopBackend:
    def test_satisfies_the_backend_protocol(self) -> None:
        assert isinstance(NoopBackend(), InputBackend)

    def test_accepts_every_command_without_complaint(self) -> None:
        backend = NoopBackend()
        dispatch(backend, [ZoomEvent(9), ScrollEvent(-9)])
        backend.close()


class TestCreateBackend:
    def test_builds_the_noop_backend_by_name(self) -> None:
        assert isinstance(create_backend("none"), NoopBackend)

    def test_rejects_an_unknown_name_and_lists_the_valid_ones(self) -> None:
        with pytest.raises(ValueError, match="unknown backend"):
            create_backend("teleport")  # type: ignore[arg-type]

    def test_every_advertised_name_is_actually_constructible(self) -> None:
        # "desktop" needs input permissions we may not have here, so it is
        # enough that it fails for an environmental reason rather than because
        # the name is unrecognised.
        for name in BACKEND_NAMES:
            with contextlib.suppress(RuntimeError):
                create_backend(name).close()


@dataclass
class FakeKeyboard:
    """Stands in for pynput's keyboard controller."""

    taps: list[Any] = field(default_factory=list)
    held: list[Any] = field(default_factory=list)
    released: list[Any] = field(default_factory=list)
    _depth: int = 0

    def tap(self, key: Any) -> None:
        assert self._depth > 0, "zoom keystrokes must be sent while the modifier is held"
        self.taps.append(key)

    def pressed(self, key: Any) -> Any:
        keyboard = self

        class _Held:
            def __enter__(self) -> None:
                keyboard.held.append(key)
                keyboard._depth += 1

            def __exit__(self, *_: object) -> None:
                keyboard._depth -= 1

        return _Held()

    def release(self, key: Any) -> None:
        self.released.append(key)


@dataclass
class FakeMouse:
    """Stands in for pynput's mouse controller."""

    scrolls: list[tuple[int, int]] = field(default_factory=list)

    def scroll(self, dx: int, dy: int) -> None:
        self.scrolls.append((dx, dy))


def make_desktop(**kwargs: Any) -> tuple[DesktopBackend, FakeKeyboard, FakeMouse]:
    """Build a DesktopBackend wired to fake controllers."""
    backend = DesktopBackend.__new__(DesktopBackend)
    keyboard, mouse = FakeKeyboard(), FakeMouse()
    backend._zoom_mode = kwargs.get("zoom_mode", "keyboard")  # type: ignore[attr-defined]
    backend._scroll_lines_per_click = kwargs.get("scroll_lines_per_click", 3)  # type: ignore[attr-defined]
    backend._keyboard = keyboard  # type: ignore[assignment]
    backend._mouse = mouse  # type: ignore[assignment]
    backend._modifier = "MODIFIER"  # type: ignore[assignment]
    backend._zoom_in_key = kwargs.get("zoom_in_key", "=")  # type: ignore[attr-defined]
    backend._zoom_out_key = kwargs.get("zoom_out_key", "-")  # type: ignore[attr-defined]
    return backend, keyboard, mouse


class TestDesktopBackendValidation:
    def test_rejects_an_unknown_zoom_mode(self) -> None:
        with pytest.raises(ValueError, match="zoom_mode must be one of"):
            DesktopBackend(zoom_mode="telekinesis")  # type: ignore[arg-type]

    def test_rejects_a_non_positive_scroll_rate(self) -> None:
        with pytest.raises(ValueError, match="scroll_lines_per_click must be at least 1"):
            DesktopBackend(scroll_lines_per_click=0)

    def test_validation_happens_before_any_input_device_is_opened(self) -> None:
        # Bad arguments must surface as ValueError even on a machine where
        # opening an input device would itself fail.
        with pytest.raises(ValueError):
            DesktopBackend(zoom_mode="nope")  # type: ignore[arg-type]


class TestDesktopKeyboardZoom:
    def test_zooming_in_taps_the_platform_zoom_in_shortcut(self) -> None:
        backend, keyboard, _ = make_desktop()
        backend.zoom(1)
        assert keyboard.taps == ["="]
        assert keyboard.held == ["MODIFIER"]

    def test_zooming_out_taps_the_platform_zoom_out_shortcut(self) -> None:
        backend, keyboard, _ = make_desktop()
        backend.zoom(-1)
        assert keyboard.taps == ["-"]

    def test_multiple_steps_become_repeated_taps_under_one_hold(self) -> None:
        # Re-pressing the modifier per step would be slower and can drop
        # keystrokes in some applications.
        backend, keyboard, _ = make_desktop()
        backend.zoom(3)
        assert keyboard.taps == ["=", "=", "="]
        assert keyboard.held == ["MODIFIER"]

    def test_multiple_negative_steps_zoom_out_repeatedly(self) -> None:
        backend, keyboard, _ = make_desktop()
        backend.zoom(-2)
        assert keyboard.taps == ["-", "-"]

    def test_a_zero_step_zoom_sends_nothing(self) -> None:
        backend, keyboard, _ = make_desktop()
        backend.zoom(0)
        assert keyboard.taps == []
        assert keyboard.held == []

    def test_keyboard_zoom_does_not_touch_the_mouse(self) -> None:
        backend, _, mouse = make_desktop()
        backend.zoom(2)
        assert mouse.scrolls == []


class TestDesktopModifierScrollZoom:
    def test_zooming_turns_the_wheel_while_the_modifier_is_held(self) -> None:
        backend, keyboard, mouse = make_desktop(zoom_mode="modifier_scroll")
        backend.zoom(2)
        assert mouse.scrolls == [(0, 2)]
        assert keyboard.held == ["MODIFIER"]
        assert keyboard.taps == []

    def test_zooming_out_turns_the_wheel_the_other_way(self) -> None:
        backend, _, mouse = make_desktop(zoom_mode="modifier_scroll")
        backend.zoom(-2)
        assert mouse.scrolls == [(0, -2)]


class TestDesktopScroll:
    def test_scrolling_up_turns_the_wheel_upward(self) -> None:
        backend, _, mouse = make_desktop()
        backend.scroll(1)
        assert mouse.scrolls == [(0, 3)]

    def test_scrolling_down_turns_the_wheel_downward(self) -> None:
        backend, _, mouse = make_desktop()
        backend.scroll(-1)
        assert mouse.scrolls == [(0, -3)]

    def test_scrolling_never_moves_the_pointer_horizontally(self) -> None:
        backend, _, mouse = make_desktop()
        backend.scroll(4)
        backend.scroll(-7)
        assert all(dx == 0 for dx, _ in mouse.scrolls)

    def test_the_lines_per_click_setting_scales_the_wheel(self) -> None:
        backend, _, mouse = make_desktop(scroll_lines_per_click=10)
        backend.scroll(2)
        assert mouse.scrolls == [(0, 20)]

    def test_a_zero_click_scroll_sends_nothing(self) -> None:
        backend, _, mouse = make_desktop()
        backend.scroll(0)
        assert mouse.scrolls == []

    def test_scrolling_does_not_hold_any_modifier(self) -> None:
        # A stray modifier would turn an ordinary scroll into a zoom.
        backend, keyboard, _ = make_desktop()
        backend.scroll(3)
        assert keyboard.held == []


class TestDesktopShutdown:
    def test_close_releases_the_modifier(self) -> None:
        # Quitting mid-zoom must not strand the modifier in a held state.
        backend, keyboard, _ = make_desktop()
        backend.close()
        assert keyboard.released == ["MODIFIER"]

    def test_close_survives_a_failing_input_device(self, monkeypatch: pytest.MonkeyPatch) -> None:
        backend, keyboard, _ = make_desktop()

        def explode(_: Any) -> None:
            raise OSError("display server went away")

        monkeypatch.setattr(keyboard, "release", explode)
        backend.close()  # must not propagate during shutdown


class TestPlatformModifier:
    def test_macos_uses_command_and_other_platforms_use_control(self) -> None:
        # Verified against the real pynput key constants rather than a fake,
        # since picking the wrong modifier silently breaks zoom everywhere.
        #
        # pynput raises at import time where there is no display server, which
        # is precisely the situation on a headless CI runner, so the import
        # itself has to be skippable rather than merely the construction.
        # exc_type is required: importorskip defaults to catching only
        # ModuleNotFoundError, and pynput is very much installed here -- it
        # raises a plain ImportError from inside itself when it cannot reach a
        # display server, which is exactly the headless CI case.
        real_keyboard = pytest.importorskip(
            "pynput.keyboard",
            reason="synthetic input needs a display server",
            exc_type=ImportError,
        )
        expected = real_keyboard.Key.cmd if sys.platform == "darwin" else real_keyboard.Key.ctrl
        try:
            backend = DesktopBackend()
        except RuntimeError:  # pragma: no cover - no input device available
            pytest.skip("no input device available in this environment")
        assert backend._modifier is expected  # type: ignore[attr-defined]


class FakeKeyCode:
    """Stands in for pynput's KeyCode, recording the virtual keycode asked for."""

    def __init__(self, vk: int) -> None:
        self.vk = vk

    def __repr__(self) -> str:
        return f"FakeKeyCode(vk={self.vk})"


class FakeKeyboardModule:
    KeyCode = type("KeyCode", (), {"from_vk": staticmethod(FakeKeyCode)})


class TestZoomKeyResolution:
    """A regression guard for zoom silently doing nothing on macOS.

    pynput resolves the *character* '=' to whichever physical key its layout
    search finds first. On macOS that is the numeric keypad's equals
    (kVK_ANSI_KeypadEquals, 81), not the main-row key (kVK_ANSI_Equal, 24).
    Applications bind zoom to the main row, so the keypad keystroke arrives
    correctly flagged with Command and is then ignored -- gestures are
    recognised, keystrokes are sent, and the document never moves.
    """

    def test_macos_names_the_main_row_equals_key_outright(self) -> None:
        key = resolve_zoom_key("=", FakeKeyboardModule, "darwin")
        assert isinstance(key, FakeKeyCode)
        assert key.vk == 24  # kVK_ANSI_Equal

    def test_macos_names_the_main_row_minus_key_outright(self) -> None:
        key = resolve_zoom_key("-", FakeKeyboardModule, "darwin")
        assert isinstance(key, FakeKeyCode)
        assert key.vk == 27  # kVK_ANSI_Minus

    def test_macos_never_resolves_a_keypad_keycode(self) -> None:
        for char in ("=", "-"):
            assert resolve_zoom_key(char, FakeKeyboardModule, "darwin").vk not in (78, 81)

    def test_macos_does_not_pass_the_bare_character_through(self) -> None:
        # Passing the character is precisely what caused the bug.
        assert resolve_zoom_key("=", FakeKeyboardModule, "darwin") != "="

    @pytest.mark.parametrize("platform", ["win32", "linux", "freebsd"])
    def test_other_platforms_keep_the_character_form(self, platform: str) -> None:
        # Unchanged where the character form is known to work; a new platform
        # only needs an entry in the keycode table.
        assert resolve_zoom_key("=", FakeKeyboardModule, platform) == "="
        assert resolve_zoom_key("-", FakeKeyboardModule, platform) == "-"

    def test_an_unmapped_character_falls_back_to_the_character(self) -> None:
        assert resolve_zoom_key("0", FakeKeyboardModule, "darwin") == "0"


class TestResolvedKeysReachTheKeyboard:
    def test_zooming_in_taps_whatever_the_resolution_produced(self) -> None:
        sentinel = FakeKeyCode(24)
        backend, keyboard, _ = make_desktop(zoom_in_key=sentinel)
        backend.zoom(1)
        assert keyboard.taps == [sentinel]

    def test_zooming_out_taps_whatever_the_resolution_produced(self) -> None:
        sentinel = FakeKeyCode(27)
        backend, keyboard, _ = make_desktop(zoom_out_key=sentinel)
        backend.zoom(-1)
        assert keyboard.taps == [sentinel]
