# Speech-to-Text-WaveNet : Deutsche
Spracherkennung auf Ende-zu-Ende-Satzebene mit DeepMind's WaveNet

Eine Tensorflow-Implementierung der Spracherkennung basierend auf DeepMinds [WaveNet: A Generative Model for Raw Audio] (https://arxiv.org/abs/1609.03499). (Im Folgenden das Paper)

Obwohl [ibab](https://github.com/ibab/tensorflow-wavenet) und [tomlepaine](https://github.com/tomlepaine/fast-wavenet) bereits WaveNet mit Tensorflow implementiert haben, haben sie keine Spracherkennung implementiert. Deshalb haben wir beschlossen, sie selbst zu implementieren. 

Einige der jüngsten Arbeiten von Deepmind sind schwer zu reproduzieren. Das Papier ließ auch spezifische Details über die Implementierung aus, und wir mussten die Lücken auf unsere eigene Weise füllen.

Hier sind ein paar wichtige Hinweise.

Erstens: Während im Paper der TIMIT-Datensatz für das Spracherkennungsexperiment verwendet wurde, haben wir den freien VTCK-Datensatz verwendet.

Zweitens fügte das Paper eine Mean-Pooling-Schicht nach der dilatierten Faltungsschicht zum Down-Sampling hinzu. Wir extrahierten [MFCC](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum) aus wav-Dateien und entfernten die letzte Mean-Pooling-Schicht, da die ursprüngliche Einstellung auf unserem TitanX-Grafikprozessor nicht lauffähig war.

Drittens: Da der TIMIT-Datensatz Phonem-Etiketten enthält, wurde das Modell in der Arbeit mit zwei Verlusttermen trainiert, der Phonem-Klassifikation und der Vorhersage des nächsten Phonems. Wir haben stattdessen einen einzigen CTC-Verlust verwendet, da VCTK Labels auf Satzebene liefert. Infolgedessen verwendeten wir nur dilated conv1d-Schichten ohne dilated conv1d-Schichten.

Schließlich haben wir aus Zeitgründen keine quantitativen Analysen wie BLEU-Score und Post-Processing durch Kombination eines Sprachmodells durchgeführt.

Die endgültige Architektur ist in der folgenden Abbildung dargestellt.
<p align="center">
  <img src="https://raw.githubusercontent.com/buriburisuri/speech-to-text-wavenet/master/png/architecture.png" width="1024"/>
</p>
(Einige Bilder sind aus [WaveNet: A Generative Model for Raw Audio] (https://arxiv.org/abs/1609.03499) und [Neural Machine Translation in Linear Time] (https://arxiv.org/abs/1610.10099) entnommen)  


## Version 

Aktuelle Version : __***0.0.0.2***__

## Abhängigkeiten ( VERSION MUSS EXAKT PASSEN! )

1. [tensorflow](https://www.tensorflow.org/versions/r0.11/get_started/os_setup.html#pip-installation) == 1.0.0
1. [sugartensor](https://github.com/buriburisuri/sugartensor) == 1.0.0.2
1. [pandas](http://pandas.pydata.org/pandas-docs/stable/install.html) >= 0.19.2
1. [librosa](https://github.com/librosa/librosa) == 0.5.0
1. [scikits.audiolab](https://pypi.python.org/pypi/scikits.audiolab)==0.11.0

If you have problems with the librosa library, try to install ffmpeg by the following command. ( Ubuntu 14.04 )  
<pre><code>
sudo add-apt-repository ppa:mc3man/trusty-media
sudo apt-get update
sudo apt-get dist-upgrade -y
sudo apt-get -y install ffmpeg
</code></pre>

## Dataset

Use [VCTK](http://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html), 
[LibriSpeech](http://www.openslr.org/12/) and [TEDLIUM release 2](http://www-lium.univ-lemans.fr/en/content/ted-lium-corpus) corpus.
Total number of sentences in the training set composed of the above three corpus is 240,612. 
Valid and test set is built using only LibriSpeech and TEDLIUM corpuse, because VCTK corpus does not have valid and test set. 
After downloading the each corpus, extract them in the 'asset/data/VCTK-Corpus', 'asset/data/LibriSpeech' and 
 'asset/data/TEDLIUM_release2' directories. 
 
Audio was augmented by the scheme in the [Tom Ko et al](http://speak.clsp.jhu.edu/uploads/publications/papers/1050_pdf.pdf)'s paper. 


## Pre-processing dataset

The TEDLIUM release 2 dataset provides audio data in the SPH format, so we should convert them to some format 
librosa library can handle. Run the following command in the 'asset/data' directory convert SPH to wave format.  
<pre><code>
find -type f -name '*.sph' | awk '{printf "sox -t sph %s -b 16 -t wav %s\n", $0, $0".wav" }' | bash
</code></pre>

If you don't have installed `sox`, please installed it first.
<pre><code>
sudo apt-get install sox
</code></pre>

We found the main bottle neck is disk read time when training, so we decide to pre-process the whole audio data into 
  the MFCC feature files which is much smaller. And we highly recommend using SSD instead of hard drive.  
  Run the following command in the console to pre-process whole dataset.
<pre><code>
python preprocess.py
</code></pre>
 

## Netzwerk trainieren

Ausführen
<pre><code>
python train.py ( <== Alle verfügbaren GPUs verwenden )
oder
CUDA_VISIBLE_DEVICES=0,1 python train.py ( <== Nur GPU 0, 1 verwenden)
</code></pre>
um das Netzwerk zu trainieren. Sie können die resultierenden ckpt-Dateien und Protokolldateien im Verzeichnis „asset/train“ sehen.
Starten Sie tensorboard --logdir asset/train/log, um den Trainingsprozess zu überwachen.

Wir haben dieses Modell auf 3 Nvidia 1080 Pascal GPUs während 40 Stunden bis 50 Epochen trainiert und wir haben die Epoche ausgewählt, in der die
Validierungsverlust ist minimal. In unserem Fall ist es Epoche 40.
Wenn Sie mit dem Fehler „Speichermangel“ konfrontiert werden, reduzieren Sie „batch_size“ in der Datei „train.py“ von 16 auf 4.

Die CTC-Verluste in jeder Epoche sind wie in der folgenden Tabelle dargestellt:

| Epoche | Zugset | gültiger Satz | Testsatz |
| :----: | ----: | ----: | ----: |
| 20 | 79,541500 | 73.645237 | 83.607269 |
| 30 | 72.884180 | 69,738348 | 80.145867 |
| 40 | 69,948266 | 66.834316 | 77.316114 |
| 50 | 69.127240 | 67,639895 | 77,866674 |


## Testen des Netzwerks

Nach Abschluss des Trainings können Sie den gültigen CTC-Verlust mit dem folgenden Befehl überprüfen oder testen.
<pre><code>
python test.py --set train|valid|test --frac 1.0(0.01~1.0)
</code></pre>
Die Option „frac“ ist nützlich, wenn Sie nur den Bruchteil des Datensatzes für eine schnelle Auswertung testen möchten.

## Sprachdatei in Deutschen Text umwandeln
 
Ausführen
<pre><code>
python detect.py --file <wave_file path>
</code></pre>
um eine Sprachdatei in den Deutschen Satz umzuwandeln. Das Ergebnis wird auf der Konsole ausgegeben.

Versuchen Sie zum Beispiel den folgenden Befehl.
<pre><code>
python detect.py --file asset/data/LibriSpeech/test-clean/1089/134686/1089-134686-0000.flac
python detect.py --file asset/data/LibriSpeech/test-clean/1089/134686/1089-134686-0001.flac
python detect.py --file asset/data/LibriSpeech/test-clean/1089/134686/1089-134686-0002.flac
python detect.py --file asset/data/LibriSpeech/test-clean/1089/134686/1089-134686-0003.flac
python detect.py --file asset/data/LibriSpeech/test-clean/1089/134686/1089-134686-0004.flac
</code></pre>

Das Ergebnis wird wie folgt aussehen:
<pre><code>
er hoffte, dass es zum Abendessen einen Platz für Rüben und Saiblinge und zerdrückte Kartoffeln und fette Hammelstücke geben würde, die in der dicken Pfefferblumen-Fatan-Sauce ausgeschöpft würden
in dich hineingestopft, riet ihm sein Bauch
nach Einbruch der frühen Nacht hüpfen hier und da die noch Lampen, woich Licht auf dem squaled Viertel der Browfles
o Berty und er Gott in deinem Kopf
numbrt tan fresh nalli wartet auf deinen kalten nit ehemann
</code></pre>

Die Grundwahrheit ist wie folgt:
<pre><code>
Er hoffte, dass es zum Abendessen einen Eintopf geben würde, Rüben und Karotten und zerquetschte Kartoffeln und fette Hammelstücke, die in einer dicken, mit Pfeffermehl gemästeten Soße ausgeschöpft würden
Stopf es in deinen Bauch, riet ihm sein Bauch
NACH FRÜHER NACHT WÜRDEN HIER UND DA DIE GELBEN LAMPEN AUFLEUCHTEN IN DEN SQUALIDEN VIERTELN DER BORDEL
HALLO BERTIE IRGENDWELCHES GUTES IN IHREM VERSTAND
NUMMER ZEHN FRESH NELLY WARTET AUF EUCH GUTE NACHT EHEMANN
</code></pre>

Wie bereits erwähnt, gibt es kein Sprachmodell, also dass es einige Fälle gibt, in denen Großbuchstaben, Satzzeichen und Wörter falsch geschrieben sind.

## Vortrainierte Modelle

Sie können eine Sprachwellendatei mit dem auf dem VCTK-Korpus vortrainierten Modell in englischen Text umwandeln.
Entpacken Sie [die folgende Zip-Datei](https://drive.google.com/file/d/0B3ILZKxzcrUyVWwtT25FemZEZ1k/view?usp=sharing&resourcekey=0-R4oPytT6GC2AGiIGi8L_ag) in das Verzeichnis 'asset/train/'.

## Docker-Unterstützung

Siehe Docker [README.md](docker/README.md).

## Zukünftige Arbeiten

1. Sprachmodell

1. Polyglotten (mehrsprachiges) Modell

Wir denken, dass wir den CTC-Beam-Decoder durch ein praktisches Sprachmodell ersetzen sollten
und das polyglotte Spracherkennungsmodell wird ein guter Kandidat für zukünftige Referenzen sein.

## Andere Ressourcen

1. [ibabs WaveNet(Sprachsynthese) Tensorflow-Implementierung](https://github.com/ibab/tensorflow-wavenet)
1. [Fast WaveNet (Sprachsynthese) Tensorflow-Implementierung von Tomlepaine] (https://github.com/ibab/tensorflow-wavenet)

## Namjus andere Repositories

1. [SugarTensor](https://github.com/buriburisuri/sugartensor)
1. [EBGAN-Tensorflow-Implementierung](https://github.com/buriburisuri/ebgan)
1. [Timeseries Gan Tensorflow-Implementierung](https://github.com/buriburisuri/timeseries_gan)
1. [Überwachte InfoGAN-Tensorflow-Implementierung] (https://github.com/buriburisuri/supervised_infogan)
1. [AC-GAN-Tensorflow-Implementierung](https://github.com/buriburisuri/ac-gan)
1. [SRGAN-Tensorflow-Implementierung](https://github.com/buriburisuri/SRGAN)
1. [ByteNet-Fast Neural Machine Translation](https://github.com/buriburisuri/ByteNet)

## Zitat

Wenn Sie diesen Code nützlich finden, zitieren Sie uns bitte in Ihrer Arbeit:

<pre><code>
Kim und Park. Speech-to-Text-WaveNet. 2016. GitHub-Repository. https://github.com/buriburisuri/.
</code></pre>

# Kontakt

Namju Kim (namju.kim@kakaocorp.com) bei KakaoBrain Corp.

Kyubyong Park (kbpark@jamonglab.com) bei KakaoBrain Corp.

Hoeun Yu (hoeuyu@ethz.ch) bei ETHZ Hörerin.
