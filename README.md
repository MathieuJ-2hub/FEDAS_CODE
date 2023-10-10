Ce projet contient une application Streamlit qui utilise un modèle RandomForest pour prédire les FEDAS CODES en fonction des caractéristiques des produits. 

Exploration des données : 
L'exploration des données a été menée dans le Jupyter notebook EDAFEDAS.ipynb. Nous avons conclus un ensemble de variables pertinentes pour entrainer un modèle de prédiction. Nous n'utilisons que des variables prétraitée afin de réduire la cardinalité de chaque variable et évité l'overfitting. 

L’entraînement de modèles de prédiction : 
Cette partie a aussi été réalisée dans le notebook. Nous avons un modèle RF qui prédit les FEDAS CODE avec une précision de 72.5%. En cas réél, je conseille d'utiliser ce modèle uniquement sur des FEDAS CODE ayant une haute fréquence dans le jeu d'entrainement. Cela permet d'augmenter la précision. 

Cas d'utilisation : 
Ce modèle pourra être pertinent dans le cas de variables à haute fréquence. Il serait jusdicieux d'associer ce modèle avec des algorithmes de préprocessing permettant de nettoyer automatique les variables d'entrainement. Par exemple avec une bibliothèque permettant de mapper les tailles pour la variable 'size' ou la variable 'color_label'. Nous pourrons rajouter un filtre sur le modèle afin de ne pas tester les produits sur lequel il a de mauvaises performances. Ce qui permettra de réduire l'effectif des personnes travaillant au mapping des FEDAS CODE.

Poursuivre la POC : 
Nous pourrons poursuivre la POC avec plus de données notamment pour les classes à un élément.

Dans le dossier build_modele
Pour la mise en production nous chargeons le modèle, le jeu de données, et faisons les prédictions dans Load_and_give_results.py
Nous pouvons rajouter une interface streamlit afin de permettre aux utilisateurs de choisir le jeu de données d'entrée, et le dossier ou exporter les résultats. 

Dans le dossier Deploy_POC : 
Nous avons un script interface utilisateur stream.py qui permet d'utiliser l'application avec une interface streamlit. 

L'application peut être testée en local en exécutant la commande : 
    streamlit run Stream.py
dans le dossier Deploy_POC. 
Mais il faut toutes les bonnes dépendances. Pour faire fonctionner l'application en local, il faut d'abord faire tourner le notebook et exporter un modèle sous le nom "rf_model.pkl" et placer ce modèle dans le fichier Deploy_POC. Initialement ce modèle devait dirèctement être incorporé dans ce repos. Mais cela a échoué à cause de sa volumétrie. 

Ainsi pour le déployement de l'application chez un client nous avons créé un Dockerfile avec un fichier requirement.txt permettant de créer une image de l'application avec toutes les dépendances et de la déployer via un conteneur chez le client. 

L'image a été mise en ligne sur Docker Hub. 

Comment utiliser l'application

Pour exécuter l'application, vous aurez besoin de Docker installé sur votre machine. Suivez les étapes ci-dessous pour télécharger l'image Docker et exécuter l'application :

    Téléchargez l'image Docker :
    Ouvrez un terminal et exécutez la commande suivante pour télécharger l'image Docker de l'application :

        docker pull matmadmax18/test_technique_mj


Exécutez l'application :
Une fois l'image Docker téléchargée, vous pouvez exécuter l'application avec la commande suivante :

    docker run -v /path/on/host:/path/in/container -p 8501:8501 matmadmax18/test_technique_mj

    notez au'il y a deux volumes car un import et un export. 


    Accédez à l'application :
    Ouvrez un navigateur web et allez à l'adresse http://localhost:8501. Vous devriez voir l'interface de l'application Streamlit.

Comment utiliser l'application:
L'application vous permet de prédire le FEDAS CODE en choissant le chemin du jeu test et elle utilise un modèle de Random Forest plus légé que dans l'EDA pour prédire les FEDAS CODE. Vous pouvez choisir les dossier d'export. 

Liste de bonnes pratiques à mettre en place pour poursuivre le développement: 
-Versionning du code avec un git repository.
-Mise en place de tests unitaire avec pytest pour les fonctions de Load_and_give_results.py
-Mise en place d'une documentation et d'une gestion des tickets pour le maintient de l'application. 
-Pour la mise en production du code, créer un Makefile.
-Qualité du code : utiliser un linter afin de garder un code propre.


