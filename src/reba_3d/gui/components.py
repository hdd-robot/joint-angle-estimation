"""
GUI components for REBA application.

Provides button, toggle, and log panel components using pygame.
"""

import pygame
from typing import Callable, Optional, List, Tuple

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
LIGHT_GRAY = (200, 200, 200)
DARK_GRAY = (60, 60, 60)
GREEN = (0, 200, 0)
RED = (200, 0, 0)
BLUE = (0, 100, 200)
YELLOW = (200, 200, 0)
ORANGE = (255, 150, 0)


class Button:
    """
    Clickable button with text.
    """

    def __init__(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        text: str,
        callback: Optional[Callable] = None,
        color: Tuple[int, int, int] = BLUE,
        hover_color: Tuple[int, int, int] = (0, 150, 255),
        text_color: Tuple[int, int, int] = WHITE,
        font_size: int = 20
    ):
        """
        Initialize a button.

        Args:
            x, y: Button position
            width, height: Dimensions
            text: Displayed text
            callback: Function called on click
            color: Background color
            hover_color: Hover color
            text_color: Text color
            font_size: Font size
        """
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.callback = callback
        self.color = color
        self.hover_color = hover_color
        self.text_color = text_color
        self.font = pygame.font.Font(None, font_size)
        self.hovered = False
        self.enabled = True

    def draw(self, surface: pygame.Surface) -> None:
        """Draw the button on the surface."""
        if not self.enabled:
            color = GRAY
        elif self.hovered:
            color = self.hover_color
        else:
            color = self.color

        pygame.draw.rect(surface, color, self.rect, border_radius=5)
        pygame.draw.rect(surface, WHITE, self.rect, 2, border_radius=5)

        text_surface = self.font.render(self.text, True, self.text_color)
        text_rect = text_surface.get_rect(center=self.rect.center)
        surface.blit(text_surface, text_rect)

    def handle_event(self, event: pygame.event.Event) -> bool:
        """
        Handle events for this button.

        Returns:
            True if the button was clicked
        """
        if not self.enabled:
            return False

        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1 and self.rect.collidepoint(event.pos):
                if self.callback:
                    self.callback()
                return True

        return False

    def set_text(self, text: str) -> None:
        """Change the button text."""
        self.text = text


class ToggleButton(Button):
    """
    Button with on/off state.
    """

    def __init__(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        text_on: str,
        text_off: str,
        callback: Optional[Callable] = None,
        color_on: Tuple[int, int, int] = GREEN,
        color_off: Tuple[int, int, int] = RED,
        initial_state: bool = False,
        **kwargs
    ):
        """
        Initialize a toggle button.

        Args:
            text_on: Text when active
            text_off: Text when inactive
            color_on: Color when active
            color_off: Color when inactive
            initial_state: Initial state
        """
        super().__init__(x, y, width, height, text_off, callback, color_off, **kwargs)
        self.text_on = text_on
        self.text_off = text_off
        self.color_on = color_on
        self.color_off = color_off
        self.state = initial_state
        self._update_appearance()

    def _update_appearance(self) -> None:
        """Update appearance based on state."""
        if self.state:
            self.text = self.text_on
            self.color = self.color_on
            self.hover_color = (0, 255, 0)
        else:
            self.text = self.text_off
            self.color = self.color_off
            self.hover_color = (255, 50, 50)

    def toggle(self) -> None:
        """Toggle the button state."""
        self.state = not self.state
        self._update_appearance()

    def set_state(self, state: bool) -> None:
        """Set the button state."""
        self.state = state
        self._update_appearance()

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Handle events with automatic toggle."""
        if not self.enabled:
            return False

        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1 and self.rect.collidepoint(event.pos):
                self.toggle()
                if self.callback:
                    self.callback(self.state)
                return True

        return False


class RadioButtonGroup:
    """
    Radio button group (only one selected at a time).
    """

    def __init__(
        self,
        x: int,
        y: int,
        options: List[str],
        callback: Optional[Callable] = None,
        button_width: int = 120,
        button_height: int = 35,
        spacing: int = 5,
        selected_color: Tuple[int, int, int] = GREEN,
        unselected_color: Tuple[int, int, int] = DARK_GRAY
    ):
        """
        Initialize a radio button group.

        Args:
            options: List of options
            callback: Function called with the selected option
        """
        self.options = options
        self.callback = callback
        self.selected_color = selected_color
        self.unselected_color = unselected_color
        self.selected_index = 0
        self.buttons: List[Button] = []

        for i, option in enumerate(options):
            btn = Button(
                x, y + i * (button_height + spacing),
                button_width, button_height,
                option,
                callback=lambda idx=i: self._select(idx),
                color=selected_color if i == 0 else unselected_color,
                font_size=18
            )
            self.buttons.append(btn)

    def _select(self, index: int) -> None:
        """Select an option."""
        self.selected_index = index
        for i, btn in enumerate(self.buttons):
            btn.color = self.selected_color if i == index else self.unselected_color
        if self.callback:
            self.callback(self.options[index])

    def get_selected(self) -> str:
        """Return the selected option."""
        return self.options[self.selected_index]

    def draw(self, surface: pygame.Surface) -> None:
        """Draw all buttons."""
        for btn in self.buttons:
            btn.draw(surface)

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Handle events for all buttons."""
        for btn in self.buttons:
            if btn.handle_event(event):
                return True
        return False


class LogPanel:
    """
    Log display panel.
    """

    def __init__(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        max_lines: int = 20,
        font_size: int = 16,
        bg_color: Tuple[int, int, int] = (30, 30, 40),
        text_color: Tuple[int, int, int] = WHITE
    ):
        """
        Initialize the log panel.

        Args:
            x, y: Position
            width, height: Dimensions
            max_lines: Maximum number of lines displayed
        """
        self.rect = pygame.Rect(x, y, width, height)
        self.max_lines = max_lines
        self.font = pygame.font.Font(None, font_size)
        self.bg_color = bg_color
        self.text_color = text_color
        self.logs: List[Tuple[str, Tuple[int, int, int]]] = []
        self.line_height = font_size + 2

    def add_log(self, message: str, color: Optional[Tuple[int, int, int]] = None) -> None:
        """
        Add a message to the log.

        Args:
            message: Message to display
            color: Message color (optional)
        """
        if color is None:
            color = self.text_color
        self.logs.append((message, color))
        if len(self.logs) > self.max_lines:
            self.logs.pop(0)

    def add_info(self, message: str) -> None:
        """Add an info message (white)."""
        self.add_log(f"[INFO] {message}", WHITE)

    def add_success(self, message: str) -> None:
        """Add a success message (green)."""
        self.add_log(f"[OK] {message}", GREEN)

    def add_warning(self, message: str) -> None:
        """Add a warning (yellow)."""
        self.add_log(f"[WARN] {message}", YELLOW)

    def add_error(self, message: str) -> None:
        """Add an error (red)."""
        self.add_log(f"[ERR] {message}", RED)

    def clear(self) -> None:
        """Clear all logs."""
        self.logs.clear()

    def draw(self, surface: pygame.Surface) -> None:
        """Draw the log panel."""
        # Background
        pygame.draw.rect(surface, self.bg_color, self.rect)
        pygame.draw.rect(surface, GRAY, self.rect, 2)

        # Title
        title_font = pygame.font.Font(None, 20)
        title = title_font.render("LOGS", True, LIGHT_GRAY)
        surface.blit(title, (self.rect.x + 10, self.rect.y + 5))

        # Messages
        y_offset = 25
        for message, color in self.logs:
            if y_offset + self.line_height > self.rect.height:
                break
            text = self.font.render(message, True, color)
            # Truncate if too long
            if text.get_width() > self.rect.width - 20:
                # Approximate truncated text
                chars = int((self.rect.width - 30) / (text.get_width() / len(message)))
                text = self.font.render(message[:chars] + "...", True, color)
            surface.blit(text, (self.rect.x + 10, self.rect.y + y_offset))
            y_offset += self.line_height


class VideoDisplay:
    """
    Video display area.
    """

    def __init__(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        bg_color: Tuple[int, int, int] = BLACK
    ):
        """
        Initialize the video display area.
        """
        self.rect = pygame.Rect(x, y, width, height)
        self.bg_color = bg_color
        self.current_frame = None
        self.overlay_scores = True
        self.scores_2d = {}
        self.scores_3d = {}
        self.risk_label = ""
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)

    def set_frame(self, frame) -> None:
        """
        Set the frame to display.

        Args:
            frame: Numpy BGR image (OpenCV format)
        """
        if frame is not None:
            import cv2
            # Convert BGR -> RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize to fit the display area
            h, w = frame_rgb.shape[:2]
            scale = min(self.rect.width / w, self.rect.height / h)
            new_w, new_h = int(w * scale), int(h * scale)
            frame_resized = cv2.resize(frame_rgb, (new_w, new_h))
            # Convert to a pygame surface
            self.current_frame = pygame.surfarray.make_surface(
                frame_resized.swapaxes(0, 1)
            )
        else:
            self.current_frame = None

    def set_scores(self, scores_2d: dict = None, scores_3d: dict = None) -> None:
        """Set the scores to display."""
        if scores_2d is not None:
            self.scores_2d = scores_2d
        if scores_3d is not None:
            self.scores_3d = scores_3d

    def set_risk_label(self, label: str) -> None:
        """Set the risk label."""
        self.risk_label = label

    def draw(self, surface: pygame.Surface) -> None:
        """Draw the video display area."""
        # Background
        pygame.draw.rect(surface, self.bg_color, self.rect)

        if self.current_frame is not None:
            # Center the frame
            frame_rect = self.current_frame.get_rect()
            frame_rect.center = self.rect.center
            surface.blit(self.current_frame, frame_rect)

            # Show scores overlay if enabled
            if self.overlay_scores:
                self._draw_scores(surface)
        else:
            # "No video" message
            text = self.font.render("No video", True, GRAY)
            text_rect = text.get_rect(center=self.rect.center)
            surface.blit(text, text_rect)

        # Border
        pygame.draw.rect(surface, GRAY, self.rect, 2)

    def _draw_scores(self, surface: pygame.Surface) -> None:
        """Draw the scores overlay."""
        y_offset = self.rect.y + 10

        # Risk label
        if self.risk_label:
            color = self._get_risk_color(self.risk_label)
            # Semi-transparent background for the label (solid black here)
            label_rect = pygame.Rect(self.rect.x + 10, y_offset - 2, 200, 25)
            pygame.draw.rect(surface, (0, 0, 0), label_rect)
            pygame.draw.rect(surface, color, label_rect, 2)
            text = self.font.render(f"REBA: {self.risk_label}", True, color)
            surface.blit(text, (self.rect.x + 15, y_offset))
            y_offset += 30

        # 3D scores
        if self.scores_3d:
            for name, value in self.scores_3d.items():
                text = self.small_font.render(f"{name}: {value:.1f}°", True, WHITE)
                surface.blit(text, (self.rect.x + 10, y_offset))
                y_offset += 16

    def _get_risk_color(self, label: str) -> Tuple[int, int, int]:
        """Return the color based on the risk level."""
        label_lower = label.lower()
        if "negligible" in label_lower or "no" in label_lower:
            return GREEN
        elif "low" in label_lower:
            return (100, 200, 255)
        elif "medium" in label_lower:
            return ORANGE
        elif "high" in label_lower:
            return RED
        elif "very" in label_lower:
            return (200, 0, 100)
        return WHITE


class ScoreButtonGroup:
    """
    Group of small horizontal buttons to select a score (0, 1, 2, 3).

    Used for REBA parameters: load, grip, activity.
    """

    def __init__(
        self,
        x: int,
        y: int,
        label: str,
        values: List[int] = None,
        callback: Optional[Callable] = None,
        initial_value: int = 0,
        button_size: int = 22,
        spacing: int = 3,
        selected_color: Tuple[int, int, int] = GREEN,
        unselected_color: Tuple[int, int, int] = (50, 50, 60),
    ):
        """
        Initialize a score button group.

        Args:
            x, y: Position (top-left corner)
            label: Label displayed to the left of the buttons
            values: List of possible values (default: [0, 1, 2, 3])
            callback: Function called with the new value
            initial_value: Initially selected value
            button_size: Size of square buttons
            spacing: Spacing between buttons
            selected_color: Color for the selected button
            unselected_color: Color for unselected buttons
        """
        self.x = x
        self.y = y
        self.label = label
        self.values = values if values is not None else [0, 1, 2, 3]
        self.callback = callback
        self.selected_value = initial_value
        self.button_size = button_size
        self.spacing = spacing
        self.selected_color = selected_color
        self.unselected_color = unselected_color

        # Fonts
        self.font = pygame.font.Font(None, 18)
        self.btn_font = pygame.font.Font(None, 20)

        # Label width
        self.label_width = 55

        # Create button rectangles
        self.buttons: List[Tuple[pygame.Rect, int]] = []
        btn_x = x + self.label_width
        for val in self.values:
            rect = pygame.Rect(btn_x, y, button_size, button_size)
            self.buttons.append((rect, val))
            btn_x += button_size + spacing

        self.hovered_index = -1

    def get_value(self) -> int:
        """Return the selected value."""
        return self.selected_value

    def set_value(self, value: int) -> None:
        """Set the selected value."""
        if value in self.values:
            self.selected_value = value

    def draw(self, surface: pygame.Surface) -> None:
        """Draw the button group."""
        # Label
        label_surface = self.font.render(self.label, True, LIGHT_GRAY)
        surface.blit(label_surface, (self.x, self.y + 6))

        # Buttons
        for i, (rect, val) in enumerate(self.buttons):
            # Color based on state
            if val == self.selected_value:
                color = self.selected_color
            elif i == self.hovered_index:
                color = (80, 80, 100)
            else:
                color = self.unselected_color

            # Button rectangle
            pygame.draw.rect(surface, color, rect, border_radius=4)
            pygame.draw.rect(surface, LIGHT_GRAY, rect, 1, border_radius=4)

            # Button text
            text = self.btn_font.render(str(val), True, WHITE)
            text_rect = text.get_rect(center=rect.center)
            surface.blit(text, text_rect)

    def handle_event(self, event: pygame.event.Event) -> bool:
        """
        Handle events.

        Returns:
            True if a value was selected
        """
        if event.type == pygame.MOUSEMOTION:
            self.hovered_index = -1
            for i, (rect, _) in enumerate(self.buttons):
                if rect.collidepoint(event.pos):
                    self.hovered_index = i
                    break

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                for rect, val in self.buttons:
                    if rect.collidepoint(event.pos):
                        self.selected_value = val
                        if self.callback:
                            self.callback(val)
                        return True

        return False


class ScoreGraph:
    """
    Real-time graph of REBA 2D and 3D scores.

    Displays two curves on the same chart:
    - Green: 3D score
    - Blue: 2D score
    """

    def __init__(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        max_points: int = 300,
        bg_color: Tuple[int, int, int] = (20, 20, 30),
        color_3d: Tuple[int, int, int] = (0, 255, 100),
        color_2d: Tuple[int, int, int] = (100, 150, 255),
    ):
        """
        Initialize the graph.

        Args:
            x, y: Graph position
            width, height: Dimensions
            max_points: Maximum number of points displayed (frames)
            bg_color: Background color
            color_3d: Color of the 3D curve
            color_2d: Color of the 2D curve
        """
        self.rect = pygame.Rect(x, y, width, height)
        self.max_points = max_points
        self.bg_color = bg_color
        self.color_3d = color_3d
        self.color_2d = color_2d

        # Score history
        self.scores_3d: List[int] = []
        self.scores_2d: List[int] = []

        # Y range (REBA scores: 1-12)
        self.y_min = 1
        self.y_max = 12

        # Internal margins for axes
        self.margin_left = 35
        self.margin_right = 10
        self.margin_top = 15
        self.margin_bottom = 20

        # Fonts
        self.font = pygame.font.Font(None, 16)
        self.title_font = pygame.font.Font(None, 18)

    def add_scores(self, score_3d: Optional[int], score_2d: Optional[int] = None) -> None:
        """
        Add a pair of scores to the history.

        Args:
            score_3d: 3D REBA score (or main score if no 2D)
            score_2d: 2D REBA score (optional, for comparison)
        """
        if score_3d is not None:
            self.scores_3d.append(score_3d)
            if len(self.scores_3d) > self.max_points:
                self.scores_3d.pop(0)

        if score_2d is not None:
            self.scores_2d.append(score_2d)
            if len(self.scores_2d) > self.max_points:
                self.scores_2d.pop(0)

    def clear(self) -> None:
        """Clear the score history."""
        self.scores_3d.clear()
        self.scores_2d.clear()

    def _value_to_y(self, value: int) -> int:
        """Convert a REBA score to a Y coordinate."""
        graph_height = self.rect.height - self.margin_top - self.margin_bottom
        # Invert Y (top = high values)
        ratio = (value - self.y_min) / (self.y_max - self.y_min)
        return int(self.rect.y + self.margin_top + graph_height * (1 - ratio))

    def _index_to_x(self, index: int, total: int) -> int:
        """Convert an index to an X coordinate."""
        graph_width = self.rect.width - self.margin_left - self.margin_right
        if total <= 1:
            return self.rect.x + self.margin_left
        ratio = index / (total - 1)
        return int(self.rect.x + self.margin_left + graph_width * ratio)

    def draw(self, surface: pygame.Surface) -> None:
        """Draw the graph."""
        # Background
        pygame.draw.rect(surface, self.bg_color, self.rect)
        pygame.draw.rect(surface, GRAY, self.rect, 1)

        # Graph drawing area
        graph_x = self.rect.x + self.margin_left
        graph_y = self.rect.y + self.margin_top
        graph_width = self.rect.width - self.margin_left - self.margin_right
        graph_height = self.rect.height - self.margin_top - self.margin_bottom

        # Horizontal grid (risk levels)
        risk_levels = [
            (1, "Negligible", (0, 100, 0)),
            (3, "Low", (100, 150, 0)),
            (6, "Medium", (200, 150, 0)),
            (10, "High", (200, 50, 0)),
            (12, "Very high", (150, 0, 50)),
        ]

        for level, label, color in risk_levels:
            y = self._value_to_y(level)
            # Grid line
            pygame.draw.line(
                surface, (50, 50, 60),
                (graph_x, y), (graph_x + graph_width, y), 1
            )
            # Y label
            text = self.font.render(str(level), True, LIGHT_GRAY)
            surface.blit(text, (self.rect.x + 5, y - 6))

        # Y axis
        pygame.draw.line(
            surface, LIGHT_GRAY,
            (graph_x, graph_y), (graph_x, graph_y + graph_height), 1
        )

        # X axis
        pygame.draw.line(
            surface, LIGHT_GRAY,
            (graph_x, graph_y + graph_height),
            (graph_x + graph_width, graph_y + graph_height), 1
        )

        # Frames label
        frames_text = self.font.render("Frames", True, LIGHT_GRAY)
        surface.blit(
            frames_text,
            (self.rect.x + self.rect.width - 50, self.rect.y + self.rect.height - 15)
        )

        # Draw the 3D curve (green)
        if len(self.scores_3d) > 1:
            points_3d = []
            for i, score in enumerate(self.scores_3d):
                x = self._index_to_x(i, len(self.scores_3d))
                y = self._value_to_y(score)
                points_3d.append((x, y))
            pygame.draw.lines(surface, self.color_3d, False, points_3d, 2)

        # Draw the 2D curve (blue)
        if len(self.scores_2d) > 1:
            points_2d = []
            for i, score in enumerate(self.scores_2d):
                x = self._index_to_x(i, len(self.scores_2d))
                y = self._value_to_y(score)
                points_2d.append((x, y))
            pygame.draw.lines(surface, self.color_2d, False, points_2d, 2)

        # Legend
        legend_x = graph_x + 10
        legend_y = self.rect.y + 3

        # 3D
        pygame.draw.line(
            surface, self.color_3d,
            (legend_x, legend_y + 5), (legend_x + 20, legend_y + 5), 2
        )
        text_3d = self.font.render("3D", True, self.color_3d)
        surface.blit(text_3d, (legend_x + 25, legend_y))

        # 2D
        pygame.draw.line(
            surface, self.color_2d,
            (legend_x + 60, legend_y + 5), (legend_x + 80, legend_y + 5), 2
        )
        text_2d = self.font.render("2D", True, self.color_2d)
        surface.blit(text_2d, (legend_x + 85, legend_y))

        # Current score
        if self.scores_3d:
            current_3d = self.scores_3d[-1]
            text = self.font.render(f"3D:{current_3d}", True, self.color_3d)
            surface.blit(text, (self.rect.x + self.rect.width - 100, legend_y))

        if self.scores_2d:
            current_2d = self.scores_2d[-1]
            text = self.font.render(f"2D:{current_2d}", True, self.color_2d)
            surface.blit(text, (self.rect.x + self.rect.width - 50, legend_y))


class RiskTimeline:
    """
    Visual timeline displaying REBA risk intervals.

    Displays a horizontal bar with colored zones based on the risk level.
    """

    def __init__(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        bg_color: Tuple[int, int, int] = (40, 40, 40)
    ):
        """
        Initialize the risk timeline.

        Args:
            x, y: Position
            width, height: Dimensions
            bg_color: Background color
        """
        self.rect = pygame.Rect(x, y, width, height)
        self.bg_color = bg_color
        self.risk_data = None
        self.total_frames = 0
        self.current_frame = 0

        # Colors per risk level (matching risk_times.json)
        self.colors = {
            "negligible risk": (0, 200, 0),      # Green
            "medium risk": (255, 150, 0),        # Orange
            "high risk": (255, 0, 0),            # Red
            "very high risk": (150, 0, 150),     # Purple
            "invalid": (128, 128, 128)           # Gray
        }

        self.font = pygame.font.Font(None, 16)

    def set_data(self, risk_data: dict, total_frames: int) -> None:
        """
        Update the risk data.

        Args:
            risk_data: Dictionary of risk intervals
            total_frames: Total number of frames
        """
        self.risk_data = risk_data
        self.total_frames = total_frames

    def set_current_frame(self, frame: int) -> None:
        """
        Set the current frame.

        Args:
            frame: Current frame number
        """
        self.current_frame = frame

    def clear(self) -> None:
        """Clear all risk timeline data."""
        self.risk_data = None
        self.total_frames = 0
        self.current_frame = 0

    def draw(self, surface: pygame.Surface) -> None:
        """Draw the timeline."""
        if not self.risk_data or self.total_frames == 0:
            # Draw nothing if there is no data
            return

        # Background
        pygame.draw.rect(surface, self.bg_color, self.rect)

        # Draw each risk interval
        for risk_level, intervals in self.risk_data.items():
            color = self.colors.get(risk_level, (255, 255, 255))

            for start_frame, end_frame in intervals:
                # Normalize positions
                x_start = self.rect.x + int((start_frame / self.total_frames) * self.rect.width)
                x_end = self.rect.x + int((end_frame / self.total_frames) * self.rect.width)

                # Ensure a minimum visible width (1 pixel)
                if x_end <= x_start:
                    x_end = x_start + 1

                # Draw the risk rectangle
                risk_rect = pygame.Rect(
                    x_start,
                    self.rect.y,
                    x_end - x_start,
                    self.rect.height
                )
                pygame.draw.rect(surface, color, risk_rect)

        # Current frame indicator (white vertical line)
        if self.current_frame > 0:
            x_current = self.rect.x + int((self.current_frame / self.total_frames) * self.rect.width)
            pygame.draw.line(
                surface,
                WHITE,
                (x_current, self.rect.y),
                (x_current, self.rect.y + self.rect.height),
                2
            )

        # Border
        pygame.draw.rect(surface, (100, 100, 100), self.rect, 2)

        # Label
        label = self.font.render("Risk timeline", True, LIGHT_GRAY)
        surface.blit(label, (self.rect.x + 5, self.rect.y - 15))


class FileSelector:
    """
    Simple file selector in Pygame.

    Displays a list of files in a directory and allows selection.
    """

    def __init__(
        self,
        directory: str = ".",
        extension: str = ".json",
        title: str = "Select a file"
    ):
        """
        Initialize the file selector.

        Args:
            directory: Starting directory
            extension: File extension to display
            title: Selection window title
        """
        import os

        self.directory = os.path.abspath(directory)
        self.extension = extension
        self.title = title
        self.selected_file = None
        self.cancelled = False

        # Scan files
        self.files = self._scan_directory()
        self.selected_index = 0
        self.scroll_offset = 0

        # UI config
        self.width = 600
        self.height = 500
        self.font = pygame.font.Font(None, 20)
        self.title_font = pygame.font.Font(None, 28)

    def _scan_directory(self) -> list:
        """Scan the directory for files with the given extension."""
        import os

        files = []

        # Add parent directory as the first option
        parent_dir = os.path.dirname(self.directory)
        if parent_dir:
            files.append((".. (Up)", parent_dir, True))

        try:
            for item in sorted(os.listdir(self.directory)):
                full_path = os.path.join(self.directory, item)

                if os.path.isdir(full_path):
                    files.append((f"📁 {item}", full_path, True))
                elif item.endswith(self.extension):
                    files.append((f"📄 {item}", full_path, False))
        except PermissionError:
            pass

        return files

    def run(self) -> Optional[str]:
        """
        Show the selector and return the selected file.

        Returns:
            Selected file path or None if cancelled
        """
        screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(self.title)
        clock = pygame.time.Clock()

        running = True
        max_visible = 15  # Number of visible files

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.cancelled = True
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.cancelled = True
                        running = False
                    elif event.key == pygame.K_UP:
                        self.selected_index = max(0, self.selected_index - 1)
                        # Adjust scroll
                        if self.selected_index < self.scroll_offset:
                            self.scroll_offset = self.selected_index
                    elif event.key == pygame.K_DOWN:
                        self.selected_index = min(len(self.files) - 1, self.selected_index + 1)
                        # Adjust scroll
                        if self.selected_index >= self.scroll_offset + max_visible:
                            self.scroll_offset = self.selected_index - max_visible + 1
                    elif event.key == pygame.K_RETURN:
                        if self.files:
                            name, path, is_dir = self.files[self.selected_index]
                            if is_dir:
                                # Navigate into directory
                                self.directory = path
                                self.files = self._scan_directory()
                                self.selected_index = 0
                                self.scroll_offset = 0
                            else:
                                # File selected
                                self.selected_file = path
                                running = False

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        # Check if click on a file
                        mouse_y = event.pos[1]
                        header_height = 60
                        item_height = 25

                        if mouse_y > header_height:
                            clicked_index = (mouse_y - header_height) // item_height + self.scroll_offset
                            if 0 <= clicked_index < len(self.files):
                                name, path, is_dir = self.files[clicked_index]
                                if is_dir:
                                    self.directory = path
                                    self.files = self._scan_directory()
                                    self.selected_index = 0
                                    self.scroll_offset = 0
                                else:
                                    self.selected_file = path
                                    running = False

                        # Cancel button
                        cancel_rect = pygame.Rect(self.width - 110, self.height - 40, 100, 30)
                        if cancel_rect.collidepoint(event.pos):
                            self.cancelled = True
                            running = False

                elif event.type == pygame.MOUSEWHEEL:
                    # Mouse wheel scroll
                    self.scroll_offset = max(0, min(
                        len(self.files) - max_visible,
                        self.scroll_offset - event.y
                    ))

            # Draw UI
            screen.fill((30, 30, 40))

            # Title
            title_surface = self.title_font.render(self.title, True, WHITE)
            screen.blit(title_surface, (10, 10))

            # Current directory
            import os
            current_dir = os.path.basename(self.directory) or self.directory
            dir_surface = self.font.render(f"Directory: {current_dir}", True, GRAY)
            screen.blit(dir_surface, (10, 40))

            # Separator line
            pygame.draw.line(screen, GRAY, (0, 60), (self.width, 60), 2)

            # File list
            y_offset = 70
            item_height = 25

            visible_files = self.files[self.scroll_offset:self.scroll_offset + max_visible]

            for i, (name, path, is_dir) in enumerate(visible_files):
                actual_index = i + self.scroll_offset

                # Background for selection
                if actual_index == self.selected_index:
                    pygame.draw.rect(
                        screen,
                        (50, 50, 100),
                        pygame.Rect(0, y_offset, self.width, item_height)
                    )

                # File/folder name
                color = LIGHT_GRAY if is_dir else WHITE
                text_surface = self.font.render(name, True, color)
                screen.blit(text_surface, (10, y_offset + 3))

                y_offset += item_height

            # Scroll indicator
            if len(self.files) > max_visible:
                scroll_bar_height = max(20, (max_visible / len(self.files)) * (max_visible * item_height))
                scroll_bar_y = 70 + (self.scroll_offset / len(self.files)) * (max_visible * item_height)
                pygame.draw.rect(
                    screen,
                    GRAY,
                    pygame.Rect(self.width - 10, scroll_bar_y, 8, scroll_bar_height),
                    border_radius=4
                )

            # Bottom separator line
            pygame.draw.line(screen, GRAY, (0, self.height - 50), (self.width, self.height - 50), 2)

            # Cancel button
            cancel_rect = pygame.Rect(self.width - 110, self.height - 40, 100, 30)
            pygame.draw.rect(screen, RED, cancel_rect, border_radius=5)
            cancel_text = self.font.render("Cancel", True, WHITE)
            screen.blit(cancel_text, (self.width - 90, self.height - 35))

            # Instructions
            help_text = self.font.render("↑↓: Navigate | Enter: Select | Esc: Cancel", True, GRAY)
            screen.blit(help_text, (10, self.height - 35))

            pygame.display.flip()
            clock.tick(30)

        return self.selected_file if not self.cancelled else None
