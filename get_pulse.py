from lib.device import Camera
from lib.processors import findFaceGetPulse # Updated import
from lib.interface import plotXY, imshow, waitKey, destroyWindow
from cv2 import moveWindow
import argparse
import numpy as np
import datetime
# Serial port code removed
import socket
import sys
from typing import Dict, List, Tuple, Optional, Any, Callable, Union

class getPulseApp:
    """
    Python application that finds a face in a webcam stream, then isolates the
    forehead.

    Then the average green-light intensity in the forehead region is gathered
    over time, and the detected person's pulse is estimated.
    """

    def __init__(self, args: argparse.Namespace):
        # Imaging device - must be a connected camera (not an ip camera or mjpeg
        # stream)
        # Serial port code removed
        self.send_udp = False

        # Setup UDP communication if requested
        udp = args.udp
        if udp:
            self.send_udp = True
            if ":" not in udp:
                ip = udp
                port = 5005
            else:
                ip, port = udp.split(":")
                port = int(port)
            self.udp = (ip, port)
            self.sock = socket.socket(socket.AF_INET,  # Internet
                                     socket.SOCK_DGRAM)  # UDP

        # Initialize cameras
        self.cameras: List[Camera] = []
        self.selected_cam = 0
        for i in range(3):
            camera = Camera(camera=i)  # first camera by default
            if camera.valid or not len(self.cameras):
                self.cameras.append(camera)
            else:
                break
                
        self.w, self.h = 0, 0
        self.pressed = 0
        
        # Initialize the pulse detector processor
        # This is designed to handle all image & signal analysis,
        # such as face detection, forehead isolation, time series collection,
        # heart-beat detection, etc.
        self.processor = findFaceGetPulse(bpm_limits=[50, 160],
                                         data_spike_limit=2500.,
                                         face_detector_smoothness=10.)

        # Init parameters for the cardiac data plot
        self.bpm_plot = False
        self.plot_title = "Data display - raw signal (top) and PSD (bottom)"

        # Maps keystrokes to specified methods
        # (A GUI window must have focus for these to work)
        self.key_controls: Dict[str, Callable] = {
            "s": self.toggle_search,
            "d": self.toggle_display_plot,
            "c": self.toggle_cam,
            "f": self.write_csv
        }

    def toggle_cam(self) -> None:
        """
        Switch to the next available camera.
        """
        if len(self.cameras) > 1:
            self.processor.find_faces = True
            self.bpm_plot = False
            destroyWindow(self.plot_title)
            self.selected_cam += 1
            self.selected_cam = self.selected_cam % len(self.cameras)

    def write_csv(self) -> None:
        """
        Writes current data to a csv file.
        """
        fn = f"Webcam-pulse{datetime.datetime.now()}"
        fn = fn.replace(":", "_").replace(".", "_")
        data = np.vstack((self.processor.times, self.processor.samples)).T
        np.savetxt(f"{fn}.csv", data, delimiter=',')
        print("Writing csv")

    def toggle_search(self) -> None:
        """
        Toggles a motion lock on the processor's face detection component.

        Locking the forehead location in place significantly improves
        data quality, once a forehead has been successfully isolated.
        """
        state = self.processor.find_faces_toggle()
        print(f"face detection lock = {not state}")

    def toggle_display_plot(self) -> None:
        """
        Toggles the data display.
        """
        if self.bpm_plot:
            print("bpm plot disabled")
            self.bpm_plot = False
            destroyWindow(self.plot_title)
        else:
            print("bpm plot enabled")
            if self.processor.find_faces:
                self.toggle_search()
            self.bpm_plot = True
            self.make_bpm_plot()
            # Position plot window near the top-left corner of the screen
            moveWindow(self.plot_title, 10, 10) 

    def make_bpm_plot(self) -> None:
        """
        Creates and/or updates the data display.
        """
        plotXY([[self.processor.times,
                self.processor.samples],
               [self.processor.freqs,
                self.processor.fft]],
              labels=[False, True],
              showmax=[False, "bpm"],
              label_ndigits=[0, 0],
              showmax_digits=[0, 1],
              skip=[3, 3],
              name=self.plot_title,
              bg=self.processor.slices[0])

    def key_handler(self) -> None:
        """
        Handle keystrokes, as set at the bottom of __init__()

        A plotting or camera frame window must have focus for keypresses to be
        detected.
        """
        self.pressed = waitKey(10) & 255  # wait for keypress for 10 ms
        if self.pressed == 27:  # exit program on 'esc'
            print("Exiting")
            for cam in self.cameras:
                cam.cam.release()
            # Serial port code removed
            sys.exit()

        for key in self.key_controls.keys():
            if chr(self.pressed) == key:
                self.key_controls[key]()

    def main_loop(self) -> None:
        """
        Single iteration of the application's main loop.
        """
        # Get current image frame from the camera
        frame = self.cameras[self.selected_cam].get_frame()
        self.h, self.w, _c = frame.shape

        # Set current image frame to the processor's input
        self.processor.frame_in = frame
        # Process the image frame to perform all needed analysis
        self.processor.run(self.selected_cam)
        # Collect the output frame for display
        output_frame = self.processor.frame_out

        # Show the processed/annotated output frame
        imshow("Processed", output_frame)
        # Ensure main window is positioned at top-left (moved after imshow)
        moveWindow("Processed", 0, 0) 

        # Create and/or update the raw data display if needed
        if self.bpm_plot:
            self.make_bpm_plot()

        # Serial port code removed

        # Send data via UDP if enabled
        if self.send_udp:
            self.sock.sendto(f"{self.processor.bpm}".encode('utf-8'), self.udp)

        # Handle any key presses
        self.key_handler()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Webcam pulse detector.')
    # Serial port arguments removed
    parser.add_argument('--udp', default=None,
                       help='udp address:port destination for bpm data')

    args = parser.parse_args()
    App = getPulseApp(args)
    while True:
        App.main_loop()
