# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# Ein Tensor ist eine allgemeine Struktur, die Skalar (0D), Vektor (1D), Matrix (2D) und höherdimensionale 
# Arrays (3D und mehr) repräsentieren kann.
# 0D: Skalar (z. B. eine einzelne Zahl)
# 1D: Vektor (z. B. eine Zahlenreihe)
# 2D: Matrix (z. B. ein Gitter von Zahlen)
# 3D und mehr: Tensoren (z. B. mehrere Matrizen gestapelt)


# %%
#Ein 3D-Tensor
import torch
tensor = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
print(tensor)

# %% [markdown]
# Ihr könnt Tensoren in PyTorch genauso manipulieren, wie ihr es mit Matrizen gewohnt seid. Die grundlegenden 
# Operationen wie Addition, Punktprodukt, und Matrixmultiplikation funktionieren auch auf Tensoren:

# %%
A = torch.tensor([[1, 2], [3, 4]])
B = torch.tensor([[5, 6], [7, 8]])
C = torch.mm(A, B)  # Matrixmultiplikation
print(C)


# %% [markdown]
# Warum Tensoren?
# Tensoren erlauben es, komplexe Modelle, die mit mehrdimensionalen Daten arbeiten (z. B. Bilder, Videos), 
# effizient zu verarbeiten. Viele moderne neuronale Netze, wie Convolutional Neural Networks (CNNs), verwenden 
# Tensoren, um mehrschichtige Matrizenoperationen durchzuführen.

# %% [markdown]
# Vergleich zu Listen 

# %% [markdown]
# Jede Liste ist ein separates Python-Objekt, und wenn du Listen verschachtelst, erzeugt Python für jede Ebene ein
# neues Objekt. Dies führt zu zusätzlichem Speicher-Overhead und langsamerem Zugriff auf Daten

# %% [markdown]
# Tensoren sind dagegen im Speicher als flache, zusammenhängende Blöcke organisiert, was den Zugriff auf die Daten
# und Berechnungen wesentlich schneller macht, besonders bei großen Datensätzen.

# %% [markdown]
# Beispiel:

# %%
# Mit Listen:
a = [[1, 2, 3], [4, 5, 6]]
b = [[2, 3, 4], [5, 6, 7]]
c = [[a[i][j] + b[i][j] for j in range(len(a[0]))] for i in range(len(a))]
print(c)
# %% [markdown]
# Das ist umständlich und ineffizient.
#
# Mit Tensoren hingegen kannst du solche Operationen direkt und mit nur einer Zeile durchführen:
#
# Mit Tensoren:
# %%
a = torch.tensor([[1, 2, 3], [4, 5, 6]])
b = torch.tensor([[1, 1, 1], [1, 1, 1]])
c = a + b  
# %% [markdown]
# Automatische Differenzierung
# Eine der Hauptstärken von Tensoren in PyTorch ist die Fähigkeit zur automatischen Differenzierung. 
# Wenn du eine komplexe Berechnung durchführen und den Gradienten für Backpropagation berechnen möchtest 
# (z.B. im maschinellen Lernen), kannst du das mit Tensoren und der Funktion requires_grad=True sehr einfach 
# machen:
# %%
x = torch.tensor(1.0, requires_grad=True) #0D Matrix die einen Wert speichert. Zudem aktivieren wir unsere Aufzeichnung für gradientenwerte
y = x**2 #Das ist nichts anderes als x = x^2
y.backward() #Berechnet unseren gradienten
print(x.grad)  # Ausgabe: 2.0
# %% [markdown]
# Ein einfaches neuronales Netz aufbauen
# Zeige, wie man mit torch.nn ein Modell erstellt. Beginnt mit einem linearen Netz und erweitert es schrittweise:
# %%
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)  
        self.fc = nn.Linear(1, 1) #Initialisierung einer ersten Schich mit Inputs und Outputs
    
    def forward(self, x): #funktion zur Forwardpropagation
        return self.fc(x) #Ausgabe
mynet1 = SimpleNN()#Das Netz Erstellen und Benutzen

ausgabe1 = mynet1(torch.tensor([5.0]))
print(ausgabe1)

# %% [markdown]
# Kurzerläuterung nochmal zu Schichten und Funktionen
# und wie können wir jetzt eine Aktivierungsfunktion auf unsere Schicht packen ganz einfach
# %%
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)  
        self.erste_Schicht = nn.Linear(1, 1) #Initialisierung einer ersten Schich
        self.aktivierung = nn.Sigmoid() # oder nn.ReLU() oder nn.Softmax() oder nn.Tanh()

    def forward(self, x): #funktion zur Forwardpropagation
        
        x = self.erste_Schicht(x) #Ausgabe

        x = self.aktivierung(x) #Anwenden der jeweiligen Aktivierungsfunktion
        
        return x

mynet1 = SimpleNN()#Das Netz Erstellen und Benutzen

ausgabe2 = mynet1(torch.tensor([5.0]))
print(ausgabe2)
# %% [markdown]
# Und wie können wir das Jetzt mit mehrern Schichten kombinieren?
# %%
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.erste_Schicht = nn.Linear(1, 10)  # Erste Schicht: 1 Eingang, 10 Ausgänge
        self.zweite_Schicht = nn.Linear(10, 1)  # Zweite Schicht: 10 Eingänge, 1 Ausgang
        self.aktivierung = nn.ReLU()  # Aktivierungsfunktion (z.B. ReLU)

    def forward(self, x):
        x = self.erste_Schicht(x)  # Erste Schicht
        x = self.aktivierung(x)     # Aktivierung nach der ersten Schicht
        x = self.zweite_Schicht(x)  # Zweite Schicht
        return x

# Das Netz erstellen und benutzen
mynet1 = SimpleNN()

# Eingabe und Ausgabe
ausgabe3 = mynet1(torch.tensor([5.0]))
print(ausgabe3)

# %% [markdown]
# Das ist jetzt alles Schön und gut aber wir haben jetzt nur im Grund das gemacht was wir in der Aufgabe aufgebaut haben
# Die Frage ist jetzt doch wie trainieren wir unser Netzwerk
# Die frage ist was soll unser Netzwerk können => lernen zahlen zu verdoppeln
# Wie definieren wir die Anzahl an Neuronen und vor allem die Anzahl der Schichten 
#
#
# Erkläre den Unterschied zwischen Adam und SGD
#
# SGD verwendet diese Art von Ableitungen direkt, um die Gewichte anzupassen.
# Der Ablauf ist wie folgt: Nach der Berechnung des Gradienten der Verlustfunktion werden die Gewichte mit einem festen 
# Lernratenwert aktualisiert
#
# Adam (Adaptive Moment Estimation):
# Adam verwendet ebenfalls diese Ableitungen, kombiniert sie jedoch mit zusätzlichen Mechanismen wie Momentum und einer 
# adaptiven Lernrate, um die Gewichtsaktualisierungen zu optimieren.
# Bei Adam werden die Gradienten, wie in der Formel gezeigt, berechnet, aber statt die Gewichte sofort nur auf Grundlage 
# der Gradienten zu aktualisieren, berechnet Adam zusätzlich:
#   - Momentum (m): Der gleitende Durchschnitt der Gradienten wird ermittelt.
#   -Adaptive Lernraten (v): Der gleitende Durchschnitt der quadrierten Gradienten wird berechnet.
#
# Zusammenfassung:
# Beide Optimierer verwenden die in der Backpropagation berechnete Ableitung (wie in der Formel gezeigt).
# SGD verwendet diese Ableitung direkt, um die Gewichte zu aktualisieren.
# Adam verwendet die Ableitung ebenfalls, kombiniert sie jedoch mit zusätzlichen Mechanismen (Momentum und adaptive Lernraten), 
# um die Aktualisierung zu optimieren.
# Die Grundidee der Gradientenberechnung bleibt bei beiden Algorithmen gleich, aber Adam ist komplexer und flexibler in der Art, 
# wie es die Gradienten und die Gewichte behandelt aber dafür auch deutlich rechenintesiver.
#
# Wir Nutzen daher unser SGD da es dementspricht was wir auf unserer Powerpoint Folie berechnet haben und für unsere 
# Zwecke locker ausreicht
#
#  
# %%
import torch
import torch.nn as nn
import torch.optim as optim # Wir holen uns diesesmal einen vordefinierter Optimiszer die in dem Framework verfügbar sind.

#Wir Bauen wieder unsere Grundstruktur

# Einfaches neuronales Netzwerk mit zwei Schichten
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.erste_Schicht = nn.Linear(1, 10)  # Erste Schicht: 1 Eingang, 10 Ausgänge
        self.zweite_Schicht = nn.Linear(10, 1)  # Zweite Schicht: 10 Eingänge, 1 Ausgang
        self.aktivierung = nn.ReLU()  # ReLU-Aktivierung für beide Schichten

    def forward(self, x):
        x = self.erste_Schicht(x)
        x = self.aktivierung(x)
        x = self.zweite_Schicht(x)
        return x

# Netz erstellen
mynet = SimpleNN()

# Optimierer (SGD) und Verlustfunktion (z.B. Mean Squared Error für Regression)
optimizer = optim.SGD(mynet.parameters(), lr=0.01)  # SGD mit fester Lernrate 0.01
loss_function = nn.MSELoss()  # Verlustfunktion: Mean Squared Error (MSE)

# Beispiel-Daten (Eingaben und Zielwerte) also unsere Trainingsdaten
inputs = torch.tensor([[1.0], [2.0], [3.0], [4.0]], requires_grad=True)  # Eingabedaten
targets = torch.tensor([[2.0], [4.0], [6.0], [8.0]])  # Zielwerte (Verdopplung der Eingaben)

# Anzahl der Trainingsdurchläufe (Epochen)
epochs = 1000

# Trainingsschleife
for epoch in range(epochs):
    # Vorwärtsdurchlauf (Forwardpropagation)
    # Die Eingaben werden durch das Netzwerk geschickt. 
    # Das Netzwerk gibt eine Vorhersage (Output) basierend auf den aktuellen Gewichten zurück.
    outputs = mynet(inputs)

    # Verlust berechnen (Differenz zwischen Vorhersage und Zielwerten)
    # Die Loss-Funktion berechnet den Fehler zwischen den Vorhersagen (outputs) und den echten Zielwerten (targets).
    # Dies ist entscheidend, um dem Netzwerk mitzuteilen, wie weit es von der richtigen Lösung entfernt ist.
    loss = loss_function(outputs, targets)

    # Gradienten zurücksetzen
    # Bevor wir den Backpropagation-Schritt durchführen, müssen wir die Gradienten vom letzten Durchlauf auf Null setzen,
    # damit sich die neuen Gradienten nicht zu den alten aufsummieren (dies könnte falsche Updates verursachen).
    optimizer.zero_grad()

    # Rückwärtsdurchlauf (Backpropagation)
    # Der Verlust wird rückwärts durch das Netzwerk propagiert, um die Gradienten der Gewichte in Bezug auf den Verlust zu berechnen.
    # Diese Gradienten werden für die Gewichtsaktualisierung verwendet.
    loss.backward()

    # Gewichte aktualisieren (SGD Schritt)
    # Der Optimierer (in diesem Fall SGD) verwendet die berechneten Gradienten, um die Gewichte zu aktualisieren und sie in Richtung 
    # der Minimierung des Fehlers zu bewegen. Dieser Schritt reduziert den Verlust im nächsten Durchlauf.
    optimizer.step()
    
    # Ausgabe alle 100 Epochen weil 100 / 100 den Restwert 0 hat und +1 weil wir bei 0 anfangen und bis 999 zählen
    if (epoch+1) % 100 == 0:
        print(f'Epoche: [{epoch+1}/{epochs}], Fehler: {loss.item():.4f}')

# Beispielausgabe nach dem Training
ausgabe = mynet(torch.tensor([5.0]))  # Test mit einem neuen Wert
print(f'Vorhersage für Eingabe 5.0: {ausgabe.item()}')

# %% [markdown]
# Speichern eines Neuronalen Netztes 

# %%
torch.save(mynet, 'mein_komplettes_netzwerk.pth')

# %% [markdown]
# Laden von unserem Netzwerk:

# %%
# Netz wieder initialisieren (muss die gleiche Architektur haben)
import torch
import torch.nn as nn
mynet = SimpleNN()

# Geladene Gewichte in das Modell laden
mynet = torch.load('mein_komplettes_netzwerk.pth', weights_only=False)

# Setzen auf "eval"-Modus, falls du es nur für Vorhersagen verwenden willst (keine Gradientenberechnungen notwendig)
mynet.eval()

# Test mit dem geladenen Netzwerk
meine_eingabe = float(input("Gebe einen Wert ein: "))
ausgabe = mynet(torch.tensor([meine_eingabe]))
print(f'Vorhersage für Eingabe {meine_eingabe}: {ausgabe.item()}')



