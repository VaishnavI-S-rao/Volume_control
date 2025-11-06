import cv2
import numpy as np
import math
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pynput.keyboard import Key, Controller

# Import mediapipe properly
try:
    import mediapipe as mp

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
except AttributeError:
    # Fallback for compatibility issues
    from mediapipe.python.solutions import hands as mp_hands
    from mediapipe.python.solutions import drawing_utils as mp_draw


class GestureAudioControl:
    def __init__(self):
        # Initialize MediaPipe Hands
        self.mp_hands = mp_hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp_draw

        # Initialize keyboard controller for media keys
        self.keyboard = Controller()

        # Initialize audio control (Windows)
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        self.volume = cast(interface, POINTER(IAudioEndpointVolume))

        # Get volume range
        self.vol_range = self.volume.GetVolumeRange()
        self.min_vol = self.vol_range[0]
        self.max_vol = self.vol_range[1]

        # Gesture states
        self.is_muted = False
        self.is_playing = True
        self.prev_gesture = None
        self.gesture_cooldown = 0

    def calculate_distance(self, p1, p2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

    def get_finger_positions(self, hand_landmarks, img_shape):
        """Get key finger landmark positions"""
        h, w, _ = img_shape

        # Thumb tip and index finger tip
        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]

        # Convert to pixel coordinates
        thumb_pos = (int(thumb_tip.x * w), int(thumb_tip.y * h))
        index_pos = (int(index_tip.x * w), int(index_tip.y * h))

        return thumb_pos, index_pos

    def detect_gesture(self, hand_landmarks):
        """Detect specific gestures"""
        # Get finger states (extended or folded)
        fingers = []

        # Thumb (special case - horizontal movement)
        if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
            fingers.append(1)
        else:
            fingers.append(0)

        # Other fingers
        tip_ids = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky
        for tip_id in tip_ids:
            if hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[tip_id - 2].y:
                fingers.append(1)
            else:
                fingers.append(0)

        # Check for thumbs up/down based on thumb and wrist position
        thumb_tip = hand_landmarks.landmark[4]
        wrist = hand_landmarks.landmark[0]
        index_mcp = hand_landmarks.landmark[5]  # Index finger base

        # Thumbs up: thumb is up, other fingers folded
        if fingers == [1, 0, 0, 0, 0] and thumb_tip.y < wrist.y:
            return "THUMBS_UP"  # Play

        # Thumbs down: thumb is down, other fingers folded
        if fingers == [1, 0, 0, 0, 0] and thumb_tip.y > index_mcp.y:
            return "THUMBS_DOWN"  # Pause

        # Gesture recognition
        if fingers == [0, 0, 0, 0, 0]:
            return "FIST"  # Mute/Unmute
        elif fingers == [1, 1, 0, 0, 0]:
            return "PINCH"  # Volume control
        elif fingers == [1, 1, 1, 1, 1]:
            return "PALM"  # Reset
        elif fingers == [0, 1, 1, 0, 0]:
            return "PEACE"  # Toggle Play/Pause
        elif fingers == [0, 1, 0, 0, 0]:
            return "POINT"  # Next track
        elif fingers == [1, 0, 0, 0, 1]:
            return "ROCK"  # Previous track

        return "NONE"

    def control_volume(self, distance, img_width):
        """Control volume based on pinch distance"""
        # Map distance to volume range
        min_distance = 30
        max_distance = 200

        # Clamp distance
        distance = max(min_distance, min(distance, max_distance))

        # Convert to volume
        vol_percent = np.interp(distance, [min_distance, max_distance], [0, 100])
        vol_value = np.interp(distance, [min_distance, max_distance], [self.min_vol, self.max_vol])

        self.volume.SetMasterVolumeLevel(vol_value, None)
        return int(vol_percent)

    def toggle_mute(self):
        """Toggle mute state"""
        self.is_muted = not self.is_muted
        self.volume.SetMute(1 if self.is_muted else 0, None)

    def toggle_play_pause(self):
        """Toggle play/pause"""
        self.is_playing = not self.is_playing
        self.keyboard.press(Key.media_play_pause)
        self.keyboard.release(Key.media_play_pause)

    def play_media(self):
        """Play media"""
        if not self.is_playing:
            self.is_playing = True
            self.keyboard.press(Key.media_play_pause)
            self.keyboard.release(Key.media_play_pause)

    def pause_media(self):
        """Pause media"""
        if self.is_playing:
            self.is_playing = False
            self.keyboard.press(Key.media_play_pause)
            self.keyboard.release(Key.media_play_pause)

    def next_track(self):
        """Skip to next track"""
        self.keyboard.press(Key.media_next)
        self.keyboard.release(Key.media_next)

    def previous_track(self):
        """Go to previous track"""
        self.keyboard.press(Key.media_previous)
        self.keyboard.release(Key.media_previous)

    def run(self):
        """Main loop"""
        cap = cv2.VideoCapture(0)
        cap.set(3, 1280)  # Width
        cap.set(4, 720)  # Height

        print("Gesture Audio Control Started!")
        print("Gestures:")
        print("- PINCH (Thumb + Index): Control Volume")
        print("- FIST: Mute/Unmute")
        print("- THUMBS UP: Play Media")
        print("- THUMBS DOWN: Pause Media")
        print("- PEACE (Index + Middle): Toggle Play/Pause")
        print("- POINT (Index only): Next Track")
        print("- ROCK (Thumb + Pinky): Previous Track")
        print("- PALM (All fingers up): Reset Volume")
        print("- Press 'q' to quit")

        while True:
            success, img = cap.read()
            if not success:
                break

            img = cv2.flip(img, 1)  # Mirror image
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.hands.process(img_rgb)

            h, w, c = img.shape
            current_volume = int(np.interp(self.volume.GetMasterVolumeLevel(),
                                           [self.min_vol, self.max_vol], [0, 100]))

            # Decrease cooldown
            if self.gesture_cooldown > 0:
                self.gesture_cooldown -= 1

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks
                    self.mp_draw.draw_landmarks(img, hand_landmarks,
                                                self.mp_hands.HAND_CONNECTIONS)

                    # Get finger positions
                    thumb_pos, index_pos = self.get_finger_positions(hand_landmarks, img.shape)

                    # Detect gesture
                    gesture = self.detect_gesture(hand_landmarks)

                    # Handle gestures
                    if gesture == "PINCH":
                        # Draw line between thumb and index
                        cv2.line(img, thumb_pos, index_pos, (0, 255, 0), 3)
                        cv2.circle(img, thumb_pos, 10, (255, 0, 255), cv2.FILLED)
                        cv2.circle(img, index_pos, 10, (255, 0, 255), cv2.FILLED)

                        # Calculate distance and control volume
                        distance = self.calculate_distance(thumb_pos, index_pos)
                        current_volume = self.control_volume(distance, w)

                    elif gesture == "FIST" and self.gesture_cooldown == 0:
                        if self.prev_gesture != "FIST":
                            self.toggle_mute()
                            self.gesture_cooldown = 15  # Cooldown frames

                    elif gesture == "PALM" and self.gesture_cooldown == 0:
                        if self.prev_gesture != "PALM":
                            # Reset to 50% volume
                            mid_vol = (self.min_vol + self.max_vol) / 2
                            self.volume.SetMasterVolumeLevel(mid_vol, None)
                            self.is_muted = False
                            self.volume.SetMute(0, None)
                            self.gesture_cooldown = 15

                    elif gesture == "PEACE" and self.gesture_cooldown == 0:
                        if self.prev_gesture != "PEACE":
                            self.toggle_play_pause()
                            self.gesture_cooldown = 15

                    elif gesture == "POINT" and self.gesture_cooldown == 0:
                        if self.prev_gesture != "POINT":
                            self.next_track()
                            self.gesture_cooldown = 15

                    elif gesture == "ROCK" and self.gesture_cooldown == 0:
                        if self.prev_gesture != "ROCK":
                            self.previous_track()
                            self.gesture_cooldown = 15

                    elif gesture == "THUMBS_UP" and self.gesture_cooldown == 0:
                        if self.prev_gesture != "THUMBS_UP":
                            self.play_media()
                            self.gesture_cooldown = 15

                    elif gesture == "THUMBS_DOWN" and self.gesture_cooldown == 0:
                        if self.prev_gesture != "THUMBS_DOWN":
                            self.pause_media()
                            self.gesture_cooldown = 15

                    self.prev_gesture = gesture

                    # Display gesture
                    cv2.putText(img, f"Gesture: {gesture}", (10, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # Draw UI
            # Volume bar
            bar_height = int(np.interp(current_volume, [0, 100], [400, 50]))
            cv2.rectangle(img, (50, 50), (85, 400), (0, 255, 0), 3)
            cv2.rectangle(img, (50, bar_height), (85, 400), (0, 255, 0), cv2.FILLED)

            # Volume percentage
            cv2.putText(img, f"Volume: {current_volume}%", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Mute status
            mute_text = "MUTED" if self.is_muted else "UNMUTED"
            mute_color = (0, 0, 255) if self.is_muted else (0, 255, 0)
            cv2.putText(img, mute_text, (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, mute_color, 2)

            # Playback status
            play_text = "PLAYING" if self.is_playing else "PAUSED"
            play_color = (0, 255, 0) if self.is_playing else (0, 165, 255)
            cv2.putText(img, play_text, (w - 250, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, play_color, 2)

            # Display
            cv2.imshow("Gesture Audio Control", img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    controller = GestureAudioControl()
    controller.run()
