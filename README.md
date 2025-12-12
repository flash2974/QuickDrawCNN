# Modèles de reconnaissance d'images Quickdraw

4 modèles différents : 
- BasicMLP : MLP (multi layer perceptron) très très simple 
- BasicCNN : CNN séquentiel avec 3 couches Conv2D superposées
- MaxPooling : BasicCNN + MaxPooling2D entre les couches de convolutions
- AVGPooling_Dropout : MaxPooling + GlobalAveragePooling2D et Dense
  

J'ai entrainé ces modèles sur une fusion de 10 datasets tirés de Google Quickdraw, à savoir :
- chien
- avion
- livre
- Tour Effiel
- voiture
- visage (smiley)
- chaise
- pomme
- cerveau
- oeil

C'est un dataset de 28x28x1 (mono-canal, en nuances de gris) donc forcément pas très performant sur des "vrais" dessins. Je rescale le dessin côté serveur puis je passe l'image rescaled aux 3 modèles du benchmark.

On dessine et ensuite les 4 prédictions sont effectuées en temps réel, et on a les prédictions avec les accuracy associées.


Pour tester : Aller sous linux (WSL ou sur windowws), et installer (avec pip): 

```
tensorflow[and-cuda]
scikit-learn
```

Puis lancer le projet sous WSL. Il vous faut aussi un dossier `quickdraw_dataset` avec dedans des datasets au format `.npy`, sur lesquels les modèles seront entrainés.

![alt text](img/image.png)