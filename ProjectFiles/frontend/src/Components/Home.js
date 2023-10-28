import { BsCloudUploadFill } from "react-icons/bs";
import "../App.css"
import axios from "axios";
import { useEffect, useState } from "react";
import aboutImage from "../media/ImageDefogger.png"
import { MdCancel } from "react-icons/md";  
import JSZip from "jszip";

function Home()
{
  const [uploadedImage, setUploadedImage] = useState([]);
  const [frontendImageLink, setFrontendImageLink] = useState([]);
  const [value, setValue] = useState("3");
  const [uploadStatus, setUploadStatus] = useState(false);
  const [uploadNextBlock, setUploadNextBlock] = useState(false);
  const [loading, setLoading] = useState(false); // New loading state
  const [defogImages, setDefogImages] = useState([])

  function handleUploadImage(event)//dataTransfer
  {
    event.preventDefault();
    const images = Array.from(event.target.files);
    if(images)
    {
      const updatedImages = images.map((image)=>URL.createObjectURL(image));
      setFrontendImageLink(prevImageLinks => [...prevImageLinks, ...updatedImages])
    }
    setUploadedImage((prevImages => [...prevImages, ...images]));
  }

  function handleDrop(event)
  {
    event.preventDefault();
    const images = Array.from(event.dataTransfer.files);
    if(images)
    {
      const updatedImages = images.map((image)=>URL.createObjectURL(image));
      setFrontendImageLink(prevImageLinks => [...prevImageLinks, ...updatedImages])
    }
    setUploadedImage((prevImages => [...prevImages, ...images]));
  }

  
  
  const handleUpload = () => {

    setUploadStatus(true);
    
  }

  useEffect(()=>{
    const TopScroll = window.pageYOffset || document.documentElement.scrollTop;
    const LeftScroll = window.pageXOffset || document.documentElement.scrollLeft;
    if(uploadStatus)
    {
      window.onscroll = function() {
        window.scrollTo(LeftScroll, TopScroll);
      };
    }
    else
    {
      window.onscroll = null;
    }
  }, [uploadStatus])


  const handleImageClick = (index)=>
  { 
    setFrontendImageLink((prevImageLinks) => {
      const updatedImageLinks = [...prevImageLinks];
      updatedImageLinks.splice(index, 1);
      return updatedImageLinks;
    });
    
    setUploadedImage((prevImages) => {
      const updatedImages = [...prevImages];
      updatedImages.splice(index, 1);
      return updatedImages;
    });

  }


  function handleRangeSlider(e)
  {
    setValue(e.target.value);
  }
  // Convert base64 image data to a valid image URL
  function getImageUrlFromBase64(base64String) {
    return `data:image/jpeg;base64,${base64String}`;
  }

  function handleUploadBtn()
  {
    
    console.log("Hit")
    if (uploadedImage) {
      const formData = new FormData();
      uploadedImage.forEach((image, index)=>{
        formData.append(`image${index}`, image)
        formData.append("value", value)
        setLoading(true);
        setUploadNextBlock(true);
        console.log("Post request initiated")
        axios.post("http://127.0.0.1:5000/image", formData)
        .then((response)=>  
        {
          setLoading(false);
          console.log(response)

          if (response.data.images && response.data.images.length > 0) {
            // Assuming the response contains an 'images' array with base64 image strings
            const imagesFromBackend = response.data.images;
  
            // Convert base64 image data to image URLs
            const imageUrls = imagesFromBackend.map(getImageUrlFromBase64);

            setDefogImages(imageUrls);
          }
          
        })
        .catch((error)=>
        {
          console.log(error)
          setLoading(false); 
        })
      })
    }
  }

  function handleCancelBtn()
  {
    setUploadStatus(false);
    if(!uploadStatus)
    {
      window.onscroll = null;
    }

    setUploadNextBlock(false);
    setLoading(false);
    setUploadedImage([]);
    setFrontendImageLink([]);

  }

  function downloadImagesAsZip(images, zipFileName) {
    const zip = new JSZip();

    // Add each image to the ZIP file
    defogImages.forEach((imageUrl, index) => {
      // Get the image filename from the URL (you can adjust this based on your actual image URLs)
      const fileName = `defogged_image_${index}.jpg`;
      fetch(imageUrl)
        .then((response) => response.blob())
        .then((blob) => {
          zip.file(fileName, blob);
          if (index === images.length - 1) {
            // After adding all images, generate the ZIP and trigger the download
            zip.generateAsync({ type: "blob" }).then((content) => {
              const url = URL.createObjectURL(content);
              const link = document.createElement("a");
              link.href = url;
              link.download = zipFileName;
              document.body.appendChild(link);
              link.click();
              document.body.removeChild(link);
              URL.revokeObjectURL(url);
            });
          }
        });
    });
  }

  function handleDownloadBtn() {
    const zipFileName = "defogged_images.zip";
    downloadImagesAsZip(defogImages, zipFileName);
  }


    return(
        <>
        <div className="mainContainer">
        <div className="mainBody">
          <h1>Image Defogging: Effortless Removal of Haze From Your Images Using AI</h1>
          <p className="pageDesc">A powerful AI-based image processing tool for removing haze from images quickly
             and easily. Our advanced algorithms utilize deep learning to identify and remove the hazy 
             pixels from your images, restoring them to crystal clear clarity. Get the best 
            results from your photos with our AI-powered haze removal technology.</p>
            
            <div className="getStartedBtn" style={{letterSpacing:"3px"}} onClick={handleUpload}>
                Upload Image
            </div>

            { uploadStatus && 
              <div className="imageOuterContainer">
                <MdCancel className="cancelBtn" size={35} onClick={handleCancelBtn} />
                <div className="imageInnerContainer">

                {loading && <div className="loadingText">Loading...</div>}
                
                  {!uploadNextBlock && <div className="imagesContainer" onDrop={(e)=>handleDrop(e)} onDragOver={(e)=>e.preventDefault()}>
                    <input type="file" multiple name="file" id="file" class="inputfile" onChange={handleUploadImage} />
                    
                    <label className="uploadContainer" for="file">
                        
                        <p style={{color:"#b2bec3"}}>Drop Images</p>
                        <p style={{color:"#b2bec3"}}>or</p>
                        <p className="chooseFileBtn">Choose a file...</p>

                    </label>
                  </div>}

                  {!uploadNextBlock && <div className="imageContainerRightBlock">
                    <div className="uploadedImageBlock">
                      {frontendImageLink && frontendImageLink.map((image, index)=>
                      {
                        return(<>
                        
                          <img
                            key={image}
                            src={image}
                            alt="Uploaded Image"
                            width={80}
                            height={80}
                            className="uploadedImages"
                            onClick={() => handleImageClick(index)}
                          />

                        </>)
                      })}
                    </div>


                    <div className="rangeSlider">
                      <input
                        type="range"
                        min="1"
                        max="4"
                        step="1"
                        value={value}
                        onChange={handleRangeSlider}
                      />
                      <div className="showSliderValue">
                          {value === "1" && "25%"}
                          {value === "2" && "50%"}
                          {value === "3" && "75%"}
                          {value === "4" && "100%"}
                      </div>
                      
                    </div>

                    <div className={`uploadBtn ${(frontendImageLink.length === 0 && value) ? "disableBtn" : ""}`} onClick={handleUploadBtn}>Upload</div>

                  </div>}
                  
                  {(!loading && uploadNextBlock)  && <div className="backendResponse">
                    
                    <h2>Defogged Images</h2>

                    <div className="responseImages">
                      {defogImages.map((imageUrl, index) => {
                        console.log(imageUrl); // Add this line to check the imageUrl being mapped
                        return (
                          <img
                            key={index}
                            src={imageUrl}
                            alt={`Defogged Image ${index}`}
                            width={100}
                            height={100}
                            className="uploadedImages"
                          />
                        );
                      })}
                    </div>
                    
                    <div className="downloadImagesBtn" onClick={handleDownloadBtn}>
                        Download Now
                    </div>

                  </div>
                  
                  }
                      
                </div>

              </div>
            }
            
        </div>

        <div className="quoteContainer">
            <p>WHAT WE BELIEVE IN</p>
            <h5>Clear Your Vision, Unleash the Beauty: Image Defogging Made Easy!</h5>
        </div>
        <h3 className="aboutHeading">ABOUT</h3>
        <div className="aboutContainer">
            <div className="aboutContect">
                <h4>What we are doing</h4>
                <p>AIMS LAB is specializing in image defogging technology for autonomous cars. Our advanced algorithms remove fog and haze, enhancing object recognition and obstacle detection. Experience safer and more reliable autonomous driving with our pixel-perfect image defogging. Drive with confidence, whatever the weather</p>
            </div>
            <div className="aboutImg">
                <img src={aboutImage} style={{width:"37rem"}}/>
            </div>
        </div>

        
        
      </div>
        </>
    )
}

export default Home;