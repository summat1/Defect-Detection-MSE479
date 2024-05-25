from util import normalize_coordinates, labeledImage 
from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateBatch, ImageFileCreateEntry, Region
from msrest.authentication import ApiKeyCredentials

class AzureCVObjectDetectionAPI(object):
    """
     A wraper class for simplifying the use of Azure Custom Vision Object Detections
    """
    

    def __init__(self, endpoint, key, resource_id, project_id=None):
        """ 
        Class Constructor, takes the id from Azure Custom Vision. Here the key will
        be used both for training and predicition
        
        Args:
        ----
        endpoint: str
        key: str
        resource_id: str
        project_id: str
        """

        training_credentials   = ApiKeyCredentials(in_headers={"Training-key": key})
        prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": key})

        # Trainer, Predictor are objects from the Azure custom vision library that 
        # we will use to upload training images or to do prediction on some images
        self.trainer   = CustomVisionTrainingClient(endpoint, training_credentials)
        self.predictor = CustomVisionPredictionClient(endpoint, prediction_credentials)
        self.project_id = project_id
        self.tags = {}

        if project_id is not None:
            for t in self.trainer.get_tags(project_id):
                self.tags[t.name] = t.id
        
        return

    def create_project(self, project_name):
        """
        Create a object detection project with name as project_name. Swith to this project
        when creation is complete.

        Args:
        ----
        project_name: str
        """
        # Find the object detection domain
        obj_detection_domain = next(domain for domain in trainer.get_domains() 
                                    if domain.type == "ObjectDetection" and domain.name == "General")

        # Create a new project
        print ("Creating project...")
        project = trainer.create_project(project_name, domain_id=obj_detection_domain.id)
        self.project_id = project.id

        return

    def create_tag(self, tag_name):
        """
        Create a tag at the current object detection project.

        Args:
        ----
        project_name: str
        """
        assert (self.project_id is not None)
        tag = self.trainer.create_tag(self.project_id, tag_name)
        self.tags[tag.name] = tag.id
        
        return

    def _upload_one_batch_training_images(self, tagged_images_with_regions):
        """
        Upload one batch (maximum 64 images, per Azure documentation) of training images to Azure Custom Vision Object Detection.
        Only for internal use with in this MSE class.
        
        Args:
        ----
        tagged_images_with_regions: list of ImageFileCreateEntry 
        
        """
        
        upload_result = self.trainer.create_images_from_files(
            self.project_id, ImageFileCreateBatch(images=tagged_images_with_regions))
        
        if not upload_result.is_batch_successful:
            print("Image batch upload failed.")
            for image in upload_result.images:
                print("Image status: ", image.status)

        return

    def upload_training_images(self, training_labeled_images):
        """
        Upload training images to Azure Custom Vision Object Detection.
        
        Args:
        ----
        training_lableded_images: list of labeledImage
        """

        # Make sure that project_id is not null or empty
        assert (self.project_id is not None)
        
        print ("Adding images...")
        tagged_images_with_regions = []
        batch = 0

        # We'll iterate through all the labeled images
        for i in range(len(training_labeled_images)):
            # We go through the labeled images in batches of 64 images max
            if i > 0 and ( i % 64 ) == 0:
                batch += 1
                print("Adding images: batch ", batch)
                # We've reached 64 images; let's upload that batch and reset "tagged_images_with_regions" for the next batch
                self._upload_one_batch_training_images(tagged_images_with_regions)
                tagged_images_with_regions = []

            # accumulating labels within one batch
            labeled_img = training_labeled_images[i]
            
            for t, labels in labeled_img.labels.items():
                
                if t not in self.tags.keys(): self.create_tag(t)

                tag_id = self.tags[t]
                
                regions = []
                for m in labels:
                    # Get the labels in a format expected by Azure Custom Visio
                    x,y,w,h = normalize_coordinates(m, labeled_img.shape)
                    regions.append(Region(tag_id=tag_id, left=x,top=y,width=w,height=h))

            # Read the actual image and create an ImageFileCreateEntry object from it for upload to Azure CV   
            with open(labeled_img.path, mode="rb") as image_contents:
                tagged_images_with_regions.append(
                    ImageFileCreateEntry(name=labeled_img.name, contents=image_contents.read(), regions=regions))
        
        batch += 1
        if len(tagged_images_with_regions) > 0: 
            print ("Adding images: batch ", batch)
            self._upload_one_batch_training_images(tagged_images_with_regions)

        return 
