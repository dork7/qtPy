import pygame 

pygame.mixer.init()
pygame.mixer.music.load("sound.wav")

def generate_audio():
	#TODO use pyDub instead
	pygame.mixer.music.play()
    while pygame.mixer.music.get_busy() == True:
       continue
