# Running the website locally

Run the following to setup jekyll

```
sudo apt-get install ruby-full build-essential zlib1g-dev

echo '# Install Ruby Gems to ~/gems' >> ~/.bashrc
echo 'export GEM_HOME="$HOME/gems"' >> ~/.bashrc
echo 'export PATH="$HOME/gems/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

gem install jekyll bundler

cd itskalvik.github.io
bundle install
```

Run the following to test the website locally

```
bundle exec jekyll serve
```