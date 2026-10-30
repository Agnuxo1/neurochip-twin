# NeuroChip Twin — presentación narrada (28-10-2026)

Abrir: `index.html` (doble clic; funciona sin internet desde esta carpeta).

## Qué es
- **Un solo bloque**: las 10 partes de la presentación, luego las 2 preguntas anticipadas (Q1, Q2) y al final «Thank you / 谢谢». Un único «Play».
- **Subtítulos bilingües**: inglés y debajo chino simplificado (traducción de cada frase; tecla Z para quitar el chino). Las preguntas aparecen arriba en inglés y chino.
- **Velocidad**: selector 1.0× / 1.1× / 1.15× / 1.2× (tono conservado). Duración total: 5:39 a 1.0×, 5:08 a 1.1×, 4:55 a 1.15×, 4:44 a 1.2× (sin contar tu presentación en directo).
- 10 animaciones de la presentación; cada una lleva su propia narración en inglés (voz ElevenLabs «Brian», la ya verificada) y subtítulos sincronizados palabra a palabra.
- 2 tarjetas animadas nuevas para el Q&A (mismo estilo que las otras 12), sincronizadas con las respuestas grabadas: cada elemento aparece cuando la voz lo menciona. En la web se reproducen una vez, enganchadas a la voz.
  - `tarjetas_qa/13_qa_innovation_with_voice.mp4` (0:37) y `tarjetas_qa/14_qa_validation_with_voice.mp4` (0:30): MP4 con la voz integrada, para usar sueltos.
  - También GIF (800×450, sin voz) y PNG del fotograma final.
- El saludo y el nombre **no** están grabados: los dices tú en directo y luego pulsas «Play narrated presentation». La narración empieza en «When we study how a substance affects…».
- Partes: presentación 4:26 · Q1 0:37 · Q2 0:30 · cierre 0:01.

## Teclas
← → parte · Espacio pausa · V voz · C subtítulos · Z chino · A avance automático · F pantalla completa · Esc cerrar.
Enlaces directos: `index.html#slide=7` (diapositiva 7), `index.html#qa=2` (respuesta 2).

## En Teams
Compartir pantalla (o la ventana del navegador) con **«Incluir sonido del equipo»** activado. Sin eso, el jurado no oye la narración. Probarlo en el ensayo técnico del 29-10.

## Cómo se hizo (reproducible)
- `work/words_cpu.py`: marcas de tiempo por palabra con Whisper medium.en en CPU (entorno `D:/AI-3D/envs/villa-voice-qa`).
- `work/build.py`: corta la narración original en el silencio más profundo entre frases (todos los cortes ≈ −82 dBFS), genera `media/audio/*.mp3`, `media/deck.js` (subtítulos, señales de imagen) y copia MP4/PNG. Sin síntesis nueva.
- `work/qa_cards.py`: dibuja las dos tarjetas con las funciones de `render_visuals.py` (paleta, cabecera, tarjetas) y los mismos ficheros de resultados (243, −10,96 %, 89,9 %); tiempos sacados de las marcas de palabra.
- `work/zh.json`: traducción al chino simplificado de cada frase narrada y de las dos preguntas.
- `work/verify_clips.py` → `work/verify_clips.json`: retranscribe cada clip; primera y última palabra completas en los 12.
- `work/render_v2.py`: vuelve a renderizar las 12 animaciones de Codex con la cabecera «NEUROCHIP TWIN V2» en `visuales_v2/` (GIF, MP4, PNG); el resto de cada imagen es idéntico píxel a píxel al original.
- Fuentes intactas en `../visuales_cientificos_20261027/english/`.
