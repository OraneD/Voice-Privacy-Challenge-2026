#!/bin/bash
#  Download exp/asv_ssl/ (embedding_model.ckpt + WavLM-Large.pt)
set -e

cd "$(dirname "$0")"
mkdir -p exp
cd exp

if [ -f asv_ssl/embedding_model.ckpt ] && [ -f asv_ssl/WavLM-Large.pt ]; then
    echo "Les deux fichiers sont déjà là."
    exit 0
fi

if [ ! -f .asv_ssl.zip ]; then
    echo "Téléchargement de asv_ssl.zip ..."
    wget -O .asv_ssl.zip https://duke.app.box.com/shared/static/wo07rwi5xcgam1v6qlzfnzb536rqug14
fi

echo "Contenu du zip :"
unzip -l .asv_ssl.zip

unzip -o .asv_ssl.zip

for f in asv_ssl/embedding_model.ckpt asv_ssl/WavLM-Large.pt; do
    if [ -f "$f" ]; then
        echo "OK : exp/$f"
    else
        echo "MANQUANT : exp/$f (regarde le contenu du zip ci-dessus)" >&2
    fi
done