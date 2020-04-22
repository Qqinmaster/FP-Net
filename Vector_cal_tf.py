import tensorflow as tf
 


class Matrix(object):
    def __init__(self, coordinates):
        self.coordinates = coordinates
        self.num_samples = tf.shape(coordinates)[0]
        self.dimension   = tf.shape(coordinates)[1]
           
    def __str__(self):
        return self.coordinates

    def plus(self,v):
        return self.coordinates + v.coordinates

    def minus(self,v):
        return self.coordinates - v.coordinates

    def magnitude(self):
        return tf.reshape(tf.sqrt(tf.reduce_sum(tf.square(self.coordinates),axis = -1)),(self.num_samples,1))

    def normalized(self):
        magnitude = self.magnitude()
        weight = tf.reshape(1.0/magnitude,(self.num_samples,1))
        return self.coordinates * weight

    def component_parallel_to(self,basis):
        u = basis.normalized()
        weight = tf.reshape(tf.reduce_sum(self.coordinates * u ,axis = -1),(self.num_samples,1))
        return u * weight

    def component_orthogonal_to(self, basis):
        projection = self.component_parallel_to(basis)
        return self.coordinates - projection
       




def NB_algorithm(original_feature,trivial_featue):
    original_feature = Matrix(original_feature)
    trivial_featue = Matrix(trivial_featue)
    d = original_feature.component_orthogonal_to(trivial_featue)
    d = Matrix(d)
    f = original_feature.component_parallel_to(d)
    return f
