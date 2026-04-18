"""Membership Inference Attacks.

Two variants per spec §6:
  - yeom.py   : threshold on victim loss/confidence (Yeom et al. 2018)
  - shokri.py : shadow-model MLP attack (Shokri et al. 2017)

Both attacks take the victim's prediction probabilities on a balanced
member/non-member evaluation set and produce (attack_accuracy, attack_auc).
"""
