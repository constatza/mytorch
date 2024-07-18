class CleanMeta(type):
    """Metaclass that wraps the fit method of a class with a clean method call.
    Ensures that the clean method is called before fitting the model.
    """

    def __new__(cls, name, bases, dct):
        original_fit = dct.get("fit")
        if original_fit:

            def fit_wrapper(self, *args, **kwargs):
                self.clean()
                return original_fit(self, *args, **kwargs)

            dct["fit"] = fit_wrapper
        return super(CleanMeta, cls).__new__(cls, name, bases, dct)


class Transformation(metaclass=CleanMeta):
    def __init__(self):
        pass

    def fit(self, train):
        raise NotImplementedError("This method needs to be overridden in subclasses.")

    def fit_transform(self, train):
        raise NotImplementedError("This method needs to be overridden in subclasses.")

    def transform(self, data):
        raise NotImplementedError("This method needs to be overridden in subclasses.")

    def inverse_transform(self, data):
        raise NotImplementedError("This method needs to be overridden in subclasses.")

    def clean(self):
        # Clean up any fitted parameters
        for attr in self.__dict__:
            setattr(self, attr, None)


class StandardScaler(Transformation):
    def __init__(self):
        super().__init__()
        self.means = None
        self.stds = None

    def fit(self, train):
        self.clean()
        self.means = train.mean(axis=(0, -1), keepdims=True)
        self.stds = train.std(axis=(0, -1), keepdims=True)
        self.stds[self.stds == 0] = 1

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def transform(self, data):
        if self.means is None or self.stds is None:
            raise RuntimeError("Scaler has not been fitted.")
        return (data - self.means) / self.stds

    def inverse_transform(self, data):
        if self.means is None or self.stds is None:
            raise RuntimeError("Scaler has not been fitted.")
        return data * self.stds + self.means


class Pipeline:
    def __init__(self, *transformations):
        self.transformations = [
            transform
            for transform in transformations
            if transform is not None and isinstance(transform, Transformation)
        ]

    def add(self, transformation):
        if not isinstance(transformation, Transformation):
            raise TypeError("All transformations must inherit from BaseTransformation.")
        self.transformations.append(transformation)

    def fit(self, train):
        for transformation in self.transformations:
            transformation.fit(train)

    def fit_transform(self, train):
        for transformation in self.transformations:
            train = transformation.fit_transform(train)
        return train

    def transform(self, data):
        for transformation in self.transformations:
            data = transformation.transform(data)
        return data

    def inverse_transform(self, data):
        for transformation in self.transformations[::-1]:
            data = transformation.inverse_transform(data)
        return data

    def clean(self):
        for transformation in self.transformations:
            transformation.clean()
