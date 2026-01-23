"""
GUI components for REBA application.

Provides button, toggle, and log panel components using pygame.
"""

import pygame
from typing import Callable, Optional, List, Tuple

# Couleurs
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
    Bouton cliquable avec texte.
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
        Initialise un bouton.

        Args:
            x, y: Position du bouton
            width, height: Dimensions
            text: Texte affiché
            callback: Fonction appelée au clic
            color: Couleur de fond
            hover_color: Couleur au survol
            text_color: Couleur du texte
            font_size: Taille de la police
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
        """Dessine le bouton sur la surface."""
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
        Gère les événements pour ce bouton.

        Returns:
            True si le bouton a été cliqué
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
        """Change le texte du bouton."""
        self.text = text


class ToggleButton(Button):
    """
    Bouton avec état on/off.
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
        Initialise un bouton toggle.

        Args:
            text_on: Texte quand activé
            text_off: Texte quand désactivé
            color_on: Couleur quand activé
            color_off: Couleur quand désactivé
            initial_state: État initial
        """
        super().__init__(x, y, width, height, text_off, callback, color_off, **kwargs)
        self.text_on = text_on
        self.text_off = text_off
        self.color_on = color_on
        self.color_off = color_off
        self.state = initial_state
        self._update_appearance()

    def _update_appearance(self) -> None:
        """Met à jour l'apparence selon l'état."""
        if self.state:
            self.text = self.text_on
            self.color = self.color_on
            self.hover_color = (0, 255, 0)
        else:
            self.text = self.text_off
            self.color = self.color_off
            self.hover_color = (255, 50, 50)

    def toggle(self) -> None:
        """Inverse l'état du bouton."""
        self.state = not self.state
        self._update_appearance()

    def set_state(self, state: bool) -> None:
        """Définit l'état du bouton."""
        self.state = state
        self._update_appearance()

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Gère les événements avec toggle automatique."""
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
    Groupe de boutons radio (un seul sélectionné à la fois).
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
        Initialise un groupe de boutons radio.

        Args:
            options: Liste des options
            callback: Fonction appelée avec l'option sélectionnée
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
        """Sélectionne une option."""
        self.selected_index = index
        for i, btn in enumerate(self.buttons):
            btn.color = self.selected_color if i == index else self.unselected_color
        if self.callback:
            self.callback(self.options[index])

    def get_selected(self) -> str:
        """Retourne l'option sélectionnée."""
        return self.options[self.selected_index]

    def draw(self, surface: pygame.Surface) -> None:
        """Dessine tous les boutons."""
        for btn in self.buttons:
            btn.draw(surface)

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Gère les événements pour tous les boutons."""
        for btn in self.buttons:
            if btn.handle_event(event):
                return True
        return False


class LogPanel:
    """
    Panneau d'affichage des logs.
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
        Initialise le panneau de logs.

        Args:
            x, y: Position
            width, height: Dimensions
            max_lines: Nombre maximum de lignes affichées
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
        Ajoute un message au log.

        Args:
            message: Message à afficher
            color: Couleur du message (optionnel)
        """
        if color is None:
            color = self.text_color
        self.logs.append((message, color))
        if len(self.logs) > self.max_lines:
            self.logs.pop(0)

    def add_info(self, message: str) -> None:
        """Ajoute un message d'information (blanc)."""
        self.add_log(f"[INFO] {message}", WHITE)

    def add_success(self, message: str) -> None:
        """Ajoute un message de succès (vert)."""
        self.add_log(f"[OK] {message}", GREEN)

    def add_warning(self, message: str) -> None:
        """Ajoute un avertissement (jaune)."""
        self.add_log(f"[WARN] {message}", YELLOW)

    def add_error(self, message: str) -> None:
        """Ajoute une erreur (rouge)."""
        self.add_log(f"[ERR] {message}", RED)

    def clear(self) -> None:
        """Efface tous les logs."""
        self.logs.clear()

    def draw(self, surface: pygame.Surface) -> None:
        """Dessine le panneau de logs."""
        # Fond
        pygame.draw.rect(surface, self.bg_color, self.rect)
        pygame.draw.rect(surface, GRAY, self.rect, 2)

        # Titre
        title_font = pygame.font.Font(None, 20)
        title = title_font.render("LOGS", True, LIGHT_GRAY)
        surface.blit(title, (self.rect.x + 10, self.rect.y + 5))

        # Messages
        y_offset = 25
        for message, color in self.logs:
            if y_offset + self.line_height > self.rect.height:
                break
            text = self.font.render(message, True, color)
            # Tronquer si trop long
            if text.get_width() > self.rect.width - 20:
                # Approximation du texte tronqué
                chars = int((self.rect.width - 30) / (text.get_width() / len(message)))
                text = self.font.render(message[:chars] + "...", True, color)
            surface.blit(text, (self.rect.x + 10, self.rect.y + y_offset))
            y_offset += self.line_height


class VideoDisplay:
    """
    Zone d'affichage vidéo.
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
        Initialise la zone d'affichage vidéo.
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
        Définit le frame à afficher.

        Args:
            frame: Image numpy BGR (OpenCV format)
        """
        if frame is not None:
            import cv2
            # Convertir BGR -> RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Redimensionner pour tenir dans la zone
            h, w = frame_rgb.shape[:2]
            scale = min(self.rect.width / w, self.rect.height / h)
            new_w, new_h = int(w * scale), int(h * scale)
            frame_resized = cv2.resize(frame_rgb, (new_w, new_h))
            # Convertir en surface pygame
            self.current_frame = pygame.surfarray.make_surface(
                frame_resized.swapaxes(0, 1)
            )
        else:
            self.current_frame = None

    def set_scores(self, scores_2d: dict = None, scores_3d: dict = None) -> None:
        """Définit les scores à afficher."""
        if scores_2d is not None:
            self.scores_2d = scores_2d
        if scores_3d is not None:
            self.scores_3d = scores_3d

    def set_risk_label(self, label: str) -> None:
        """Définit le label de risque."""
        self.risk_label = label

    def draw(self, surface: pygame.Surface) -> None:
        """Dessine la zone vidéo."""
        # Fond
        pygame.draw.rect(surface, self.bg_color, self.rect)

        if self.current_frame is not None:
            # Centrer le frame
            frame_rect = self.current_frame.get_rect()
            frame_rect.center = self.rect.center
            surface.blit(self.current_frame, frame_rect)

            # Afficher les scores si activé
            if self.overlay_scores:
                self._draw_scores(surface)
        else:
            # Message "pas de vidéo"
            text = self.font.render("Pas de video", True, GRAY)
            text_rect = text.get_rect(center=self.rect.center)
            surface.blit(text, text_rect)

        # Bordure
        pygame.draw.rect(surface, GRAY, self.rect, 2)

    def _draw_scores(self, surface: pygame.Surface) -> None:
        """Dessine les scores en overlay."""
        y_offset = self.rect.y + 10

        # Risk label
        if self.risk_label:
            color = self._get_risk_color(self.risk_label)
            # Fond semi-transparent pour le label
            label_rect = pygame.Rect(self.rect.x + 10, y_offset - 2, 200, 25)
            pygame.draw.rect(surface, (0, 0, 0), label_rect)
            pygame.draw.rect(surface, color, label_rect, 2)
            text = self.font.render(f"REBA: {self.risk_label}", True, color)
            surface.blit(text, (self.rect.x + 15, y_offset))
            y_offset += 30

        # Scores 3D
        if self.scores_3d:
            for name, value in self.scores_3d.items():
                text = self.small_font.render(f"{name}: {value:.1f}°", True, WHITE)
                surface.blit(text, (self.rect.x + 10, y_offset))
                y_offset += 16

    def _get_risk_color(self, label: str) -> Tuple[int, int, int]:
        """Retourne la couleur selon le niveau de risque."""
        label_lower = label.lower()
        if "negligible" in label_lower or "sans" in label_lower:
            return GREEN
        elif "low" in label_lower or "faible" in label_lower:
            return (100, 200, 255)
        elif "medium" in label_lower or "moyen" in label_lower:
            return ORANGE
        elif "high" in label_lower or "élevé" in label_lower:
            return RED
        elif "very" in label_lower or "très" in label_lower:
            return (200, 0, 100)
        return WHITE
