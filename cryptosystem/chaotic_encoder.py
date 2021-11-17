#!/usr/bin/env python3
import gi
import threading
import subprocess, os
import concurrent.futures

# Requirements
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk, GLib

# Defaults
settings = Gtk.Settings.get_default()
settings.set_property("gtk-application-prefer-dark-theme", True)

# Libraries
import cryptosystem as crp


class ShowImageWindow(object):
    def __init__(self, image_name):
        self.window = Gtk.Window()
        self.window.add(Gtk.Image.new_from_file(image_name))
        self.window.set_title("View Image")
        self.window.set_size_request(300, 200)
        self.window.show_all()


class MainWindow:
    def __init__(self):
        builder = Gtk.Builder()
        builder.add_from_file("CryptoEncoder.glade")
        self.main_window = builder.get_object("MainWindow")
        self.view_image_button = builder.get_object("VIEW_IMAGE_BUTTON")
        self.encode_button = builder.get_object("ENCODE_BUTTON")
        self.decode_button = builder.get_object("DECODE_BUTTON")
        self.get_image_button = builder.get_object("GET_IMAGE_BUTTON")
        self.image_window = builder.get_object("IMAGE_WINDOW")
        self.progressbar = builder.get_object("PROGRESSBAR")
        builder.connect_signals(self)
        self.main_window.show_all()

        Gtk.main()

    def view_image_button_clicked(self, *args):
        print("View Image Button Clicked")
        print(self.get_image_button.get_filename())
        im_win = ShowImageWindow(
            image_name=self.get_image_button.get_filename()
        )

    def encode_button_clicked(self, *args):
        print("Encode Button Clicked")
        encrypted_image = crp.encrypt_image(
            self.get_image_button.get_filename(), self.progressbar
        )
        self.progressbar.set_fraction(0.0)

    def decode_button_clicked(self, *args):
        print("Decode Button Clicked")
        decrypted_image = crp.decrypt_image(
            self.get_image_button.get_filename(), self.progressbar
        )
        self.progressbar.set_fraction(0.0)

    def on_main_window_destroy(self, *args):
        print("Closing Application Safely")
        Gtk.main_quit()


try:
    MainWindow()

except:
    raise
    sys.exit(0)
