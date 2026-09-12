simd_8_bit.cpp e simd_32_bit.cpp
contengono rispettivamente l'implementazione dell'algoritmo
di ordinamento SIMD con istruzioni Intel SSE2 per valori a 
8 bit e valori a 32 bit.


simd_utils.h contiene funzioni utili per il debug e per la
stampa dei valori.

logicaBinaria.txt è un file che contiene la spiegazione di come
sono usate le maschere


Si consiglia la visione di image.png, la quale spiega in modo esaustivo
l'idea dietro l'algoritmo, prima di leggere il codice.
Oltretutto la versione con 32 bit risulta più intuitiva da capire,
sarebbe quindi da visualizzare prima della versione a 8 bit, la quale
invece contiene molte istruzioni "di circostanza", ovvero dei tecnicismi
obbligatori per la natura dei dati trattati, non inerenti al paradigma
di ordinamento vero e proprio. 

