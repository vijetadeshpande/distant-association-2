# 1. Download the specific linux-amd64 zip package from rclone downloads
apt install zip -y
curl -O https://downloads.rclone.org/rclone-current-linux-amd64.zip

# 2. Extract the package
unzip rclone-current-linux-amd64.zip

# 3. Create a local bin inside your home directory 
mkdir -p ~/.local/bin

# 4. Move the standalone executable into your local directory
cp rclone-*-linux-amd64/rclone ~/.local/bin/

# 5. Clean up downloaded zip files to keep cluster storage clean
rm -rf rclone-*-linux-amd64*

# 6. Ensure the local bin folder is inside your shell profile PATH
if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
    source ~/.bashrc
fi
