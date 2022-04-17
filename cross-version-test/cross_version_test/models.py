import typing as t

from pydantic import BaseModel


class PackageInfo(BaseModel):
    pip_release: str
    install_dev: t.Optional[str]


class Category(BaseModel):
    minimum: str
    maximum: str
    requirements: t.Optional[t.Dict[str, t.List[str]]]
    unsupported: t.Optional[t.List[str]]
    run: str


class Flavor(BaseModel):
    package_info: PackageInfo
    models: t.Optional[Category]
    autologging: t.Optional[Category]

    @property
    def categories(self) -> t.Dict[str, t.Optional[Category]]:
        return {
            "models": self.models,
            "autologging": self.autologging,
        }


class Job(BaseModel):
    group: str
    category: str
    flavor: str
    install: str
    run: str
    package: str
    version: str
    supported: bool

    @property
    def name(self) -> str:
        return "_".join(map(str, [self.group, self.version, self.category]))

    def __hash__(self) -> int:
        return hash(frozenset(self.dict()))
