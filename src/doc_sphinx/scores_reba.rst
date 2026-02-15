===================
Scores REBA manuels
===================

Certains paramètres REBA ne peuvent pas être détectés automatiquement par la caméra
et doivent être renseignés manuellement par l'opérateur. Ces paramètres sont
accessibles via les boutons de score dans l'interface graphique.

Vue d'ensemble
==============

+----------------+---------------------------+-------------------+
| Paramètre      | Nom dans l'interface      | Plage de valeurs  |
+================+===========================+===================+
| Charge/Force   | **Poids**                 | 0, 1, 2, 3        |
+----------------+---------------------------+-------------------+
| Qualité prise  | **Prise**                 | 0, 1, 2, 3        |
+----------------+---------------------------+-------------------+
| Activité       | **Activité**              | 0, 1, 2, 3        |
+----------------+---------------------------+-------------------+

.. important::
   Ces scores s'ajoutent au score REBA calculé automatiquement et peuvent
   significativement augmenter le niveau de risque final.

Score Charge/Force (Poids)
==========================

Évalue la charge manipulée par le travailleur.

+-------+------------------------+-------------------------------------------+
| Score | Charge                 | Description                               |
+=======+========================+===========================================+
| **0** | < 5 kg                 | Charge légère ou pas de charge            |
+-------+------------------------+-------------------------------------------+
| **1** | 5 - 10 kg              | Charge modérée                            |
+-------+------------------------+-------------------------------------------+
| **2** | > 10 kg                | Charge lourde                             |
+-------+------------------------+-------------------------------------------+
| **3** | Choc ou force soudaine | Ajout de +1 si application brusque        |
|       |                        | de force ou choc                          |
+-------+------------------------+-------------------------------------------+

Exemples pratiques
------------------

- **Score 0** : Travail sur clavier, manipulation de petits objets
- **Score 1** : Port d'une caisse de 7 kg, utilisation d'un outil de 6 kg
- **Score 2** : Déplacement d'un carton de 15 kg, port d'une charge lourde
- **Score 3** : Rattraper une charge qui tombe, coup de marteau, poussée brusque

.. note::
   Le score 3 correspond au score 2 + malus de choc (+1).
   Il s'applique quand la force est appliquée de manière soudaine ou répétée.

Score Qualité de Prise (Prise)
==============================

Évalue la qualité de la préhension sur l'objet manipulé.

+-------+------------------+------------------------------------------------+
| Score | Qualité          | Description                                    |
+=======+==================+================================================+
| **0** | Bonne prise      | Poignée bien conçue, prise en force possible   |
+-------+------------------+------------------------------------------------+
| **1** | Prise acceptable | Prise acceptable mais non idéale               |
+-------+------------------+------------------------------------------------+
| **2** | Mauvaise prise   | Prise difficile, pas de poignée                |
+-------+------------------+------------------------------------------------+
| **3** | Inacceptable     | Prise dangereuse, objet glissant ou instable   |
+-------+------------------+------------------------------------------------+

Critères d'évaluation
---------------------

**Score 0 - Bonne prise :**

- Poignée ergonomique
- Surface antidérapante
- Prise en main complète possible
- Diamètre de poignée adapté (3-4 cm)

**Score 1 - Prise acceptable :**

- Poignée présente mais non optimale
- Prise possible mais inconfortable sur la durée
- Surface légèrement glissante

**Score 2 - Mauvaise prise :**

- Absence de poignée dédiée
- Prise par les bords ou coins
- Nécessite une prise en pince prolongée
- Objet encombrant difficile à saisir

**Score 3 - Prise inacceptable :**

- Objet glissant (huileux, mouillé)
- Forme instable ou déséquilibrée
- Nécessite des gants épais réduisant la sensibilité
- Risque de lâcher l'objet

Exemples pratiques
------------------

- **Score 0** : Valise avec poignée ergonomique, outil avec manche adapté
- **Score 1** : Carton avec découpes pour les mains, bouteille standard
- **Score 2** : Plaque de métal lisse, sac souple sans poignée
- **Score 3** : Pièce huileuse, charge instable, objet très chaud/froid

Score Activité
==============

Évalue les conditions d'activité qui augmentent le risque musculo-squelettique.

+-------+--------------------------------------------------+
| Score | Condition                                        |
+=======+==================================================+
| **0** | Aucune condition aggravante                      |
+-------+--------------------------------------------------+
| **+1**| Posture statique maintenue > 1 minute            |
+-------+--------------------------------------------------+
| **+1**| Mouvements répétitifs (> 4 fois/minute)          |
+-------+--------------------------------------------------+
| **+1**| Changements posturaux rapides ou instables       |
+-------+--------------------------------------------------+

.. important::
   Le score activité est **cumulatif**. Si plusieurs conditions sont présentes,
   additionnez les points.

Détail des conditions
---------------------

**Posture statique > 1 minute (+1)**

Le travailleur maintient la même posture pendant plus d'une minute :

- Position penchée prolongée
- Bras en élévation maintenue
- Position accroupie statique

**Mouvements répétitifs (+1)**

Actions répétées plus de 4 fois par minute :

- Gestes de vissage/dévissage répétés
- Manipulation répétitive sur chaîne
- Frappes répétées (clavier intense, martelage)

**Changements posturaux rapides (+1)**

Instabilité ou mouvements brusques :

- Se lever/s'asseoir fréquemment
- Rotations rapides du tronc
- Mouvements sur surface instable
- Actions nécessitant de l'équilibre

Exemples de calcul
------------------

**Exemple 1 : Opérateur sur chaîne de montage**

- Gestes répétitifs toutes les 10 secondes → +1
- Position debout statique → +1
- Total : **Score 2**

**Exemple 2 : Manutentionnaire**

- Postures variées, pas statique → 0
- Pas de répétitivité → 0
- Mouvements stables → 0
- Total : **Score 0**

**Exemple 3 : Peintre en bâtiment**

- Bras en élévation > 1 min → +1
- Mouvements répétitifs de pinceau → +1
- Position sur échelle (instable) → +1
- Total : **Score 3**

Impact sur le score final
=========================

Ces trois scores manuels s'ajoutent au score REBA automatique selon la formule :

.. code-block:: text

   Score A' = Score A + Poids
   Score B' = Score B + Prise
   Score C  = Table C (Score A', Score B')
   Score Final = Score C + Activité

Exemple de calcul complet
-------------------------

1. Score A (tronc + cou + jambes) calculé automatiquement : **4**
2. Score B (bras + coude + poignet) calculé automatiquement : **3**
3. Poids (charge de 8 kg) : **1**
4. Prise (carton sans poignée) : **2**
5. Activité (mouvements répétitifs) : **1**

Calcul :

- Score A' = 4 + 1 = **5**
- Score B' = 3 + 2 = **5**
- Score C (Table C[5,5]) = **6**
- Score Final = 6 + 1 = **7** → high risk

Recommandations
===============

Pour minimiser les scores manuels :

**Réduire la charge (Poids)**

- Utiliser des aides mécaniques (chariot, palan)
- Fractionner les charges lourdes
- Travailler en binôme

**Améliorer la prise**

- Ajouter des poignées ergonomiques
- Utiliser des gants adaptés
- Éviter les surfaces glissantes

**Limiter l'activité à risque**

- Alterner les tâches
- Introduire des micro-pauses
- Stabiliser les surfaces de travail
