# rcv_sampling.py

## Überblick
Erzeugt ein RCV-Datenset (Recaptured Control Variation) ohne die angegebenen IDs.
Durch das Entfernen bestimmter Beobachtungen entsteht ein neutrales Vergleichsset. Es kann zum Benchmarking oder zur Kontrolle von Klassifikatoren eingesetzt werden. Die Methode reduziert jedoch die verfügbare Datenmenge.

## Wichtige Bestandteile
- generate_rcv

## Referenzen
- https://pandas.pydata.org/docs/

## Annahmen und Einschränkungen
Ersetzt Bezeichner durch "RCV" zur Verwendung als Referenz.
