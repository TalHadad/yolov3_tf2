# raspberry_pi_controller.py

import RPi.GPIO as GPIO
# define pins numbering scheme (physical (board) or gpio (bcm))
GPIO.setmode(GPIO.BOARD)
# (or) GPIO.setmode(GPIO.BCM)

# define output pins
GPIO.setup(11, GPIO.OUT)

# send signal throght output pin
GPIO.output(11, True) # or 1 instead of True
#GPIO.output(11, False) # or 0 instead of False

# release definition from pins
GPIO.cleanup()