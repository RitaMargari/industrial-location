# industrial-location

### About

1: Сервис оценки потенциала городов для развития промышленной отрасли с точки зрения обеспеченности трудовыми ресурсами

2: Сервиса оценки качества жизни специалистов в городе

### Backend

Сервис предоставляется через docker container с API:

```shell
docker pull oneonwar/industrial_location:3.2
docker container run oneonwar/industrial_location:3.2
```

### Example

В папке example_notebooks есть ноутбук с примером вызова функций, отвечающих за
функционал API

### Cite
```shell
@software{industrial-location,
  author = {Mishina M., Kharlov L., Kontsevik G.},
  title = {{industrial-location}},
  url = {https://github.com/RitaMargari/industrial-location},
  version = 3.2},
  year = {2023}
}

@article{kontsevik2023estimating,
  title={Estimating the attractiveness of the city for skilled workers using jobs-housing matching, spatial data and NLP techniques},
  author={Kontsevik, Georgii I and Zakharenko, Nikita N and Budennyy, Semen A and Mityagin, Sergey A},
  journal={Procedia Computer Science},
  volume={229},
  pages={188--197},
  year={2023},
  publisher={Elsevier}
}
```
