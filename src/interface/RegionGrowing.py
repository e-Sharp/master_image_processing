import cv2
import numpy as np
import colorsys

class RegionGrowing:
    def __init__(self, image, mask, seeds):
        self.image = image
        self.mask = mask

        self.width = image.shape[1]
        self.height = image.shape[0]

        print(self.width, self.height)

        self.indexing = {0: None}
        self.regions = np.zeros(image.shape[:2], np.uint8)

        self.queue = []
        for i, (group, seeds) in enumerate(seeds.items()):
            for (x, y) in seeds:
                j = 120 * (i + 1)
                self.indexing[group] = j
                self.queue.append(((x, y), j))
                self.regions[y, x] = j
            
    def grow(self):
        next_queue = []
        for (position, region) in self.queue:
            for (x, y) in self.blank_neighbors(position):
                self.regions[y, x] = region
                next_queue.append(((x, y), region))
        self.queue = next_queue
        return len(self.queue)

    def index(self, region):
        self.indexing[region]

    def neighbors(self, position):
        (x, y) = position
        ns = []
        if x > 0:
            ns.append((x - 1, y))
        if x < self.width - 1:
            ns.append((x + 1, y))
        if y > 0:
            ns.append((x, y - 1))
        if y < self.height - 1:
            ns.append((x, y + 1))
        return ns

    def result(self):
        return self.regions

    def saturate(self):
        while self.grow(): None

    def unmasked_neighbors(self, position):
        return [(x, y) for (x, y) in self.neighbors(position) if self.mask[y, x] != 0]

    def blank_neighbors(self, position):
        return [(x, y) for (x, y) in self.unmasked_neighbors(position) if self.regions[y, x] == 0]

    def image_representation(self):
        def hsv2rgb(h, s, v):
            return tuple(round(i * 255.) for i in colorsys.hsv_to_rgb(h, s, v))

        maxRegion = 6
        div = 1. / maxRegion

        resImage = np.zeros(self.image.shape)

        for x in range(len(self.regions)):
            for y in range(len(self.regions[0])):
                resImage[x][y] = hsv2rgb(div*self.regions[x][y], 1, 1) if self.regions[x][y] else (0,0,0)

        return resImage
