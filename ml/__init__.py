"""InondaVision CI — moteur Deep Learning de segmentation des inondations.

Classes de sortie (3 classes) :
    0 = background        (terre non inondée)
    1 = permanent_water   (lagunes, cours d'eau, plans d'eau permanents)
    2 = flood_water       (eau temporaire liée à une inondation)

Entrée standard : Sentinel-1 pré/post-événement, 2 canaux VV/VH en dB,
512x512 ou tuiles réelles de n'importe quelle taille (inférence par fenêtres).
"""

__version__ = "0.1.0"
