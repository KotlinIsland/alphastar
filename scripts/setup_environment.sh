if [ -d ./starcraft ]; then
  exit
fi

mkdir starcraft
cd starcraft || exit 1
pwd
ls -la
echo "========================================================"
wget https://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip
unzip -P iagreetotheeula SC2.4.10.zip
echo "========================================================"
rm SC2.4.10.zip
pwd
ls -la

cd ..
cp maps/* starcraft/StarCraftII/Maps/
