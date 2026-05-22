---
trigger: always_on
---

No python packages are installed locally. So tests or other commands that depend upon packages won't work. Use docker to run such things. Each repository should have a DockerFile and in some cases a docker-compose.yml. That will give you an idea on which image to use.