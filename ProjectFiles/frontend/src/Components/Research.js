import R1 from "../media/RE1.png"
import R2 from "../media/RE2.png"
import R3 from "../media/RE3.png"
import R4 from "../media/RE4.png"

function Research()
{
    return(
        <>
            <div className="researchContainer">
                <div className="reasearchInnerContainer">
                    <h1>Our Research</h1>
                </div>
            </div>

            <div className="researchQuote">
                <p>OUR TEAM DID EXTENSIVE RESEARCH ON THE TOPIC TO FIND THE VARIOUS TECHNIQUES THAT COULD BE USED AND THE HAVE A BETTER UNDERSTANDING OF WHAT <br></br>WE ARE DEALING WITH.</p>
            </div>

            <div className="ourFindings">
                <h1>Our Findings</h1>
                <img src={R1}/>
                <div className="r1">
                    <h4><a href="https://ieeexplore.ieee.org/abstract/document/6898864" target="_blank">Single Image Defogging by Multiscale Depth Fusion</a></h4>
                    <p>This algorithm utilizes an edge-preserving regularization technique to compute a regularized depth map. It employs alternate optimization methods to estimate the depth map and line field, addressing the computational challenges. The graph-cut method is used for efficient energy minimization, ensuring convergence. Additionally, noise variances are estimated using a data-driven approach. Finally, the algorithm incorporates techniques such as noise and artifact removal, depth map fusion, and depth-guided image enhancement to improve the overall quality of the image.</p>
                </div>
                <img src={R2}/>
                <div className="r2">
                    <h4><a href="https://ieeexplore.ieee.org/abstract/document/9511329" target="_blank">An Adaptive Defogging Intensity Parameter Model</a></h4>
                    <p>The Two-Dimensional Otsu algorithm is commonly used for automatic thresholding in image segmentation. It helps separate foreground and background regions by finding an optimal threshold value. The Mixed Dark Channel algorithm enhances foggy images by segmenting them and adjusting dark channels based on fog density. An Adaptive Defogging Intensity Parameter Model dynamically adjusts the fogging investigative parameter to improve defogging, with the optimal value varying for different images. The model also estimates atmospheric light and computes the transmission map to achieve better restoration.</p>
                </div>

                <img src={R3}/>
                <div className="r3">
                    <h4><a href="https://ieeexplore.ieee.org/abstract/document/9511329" target="_blank">An Adaptive Defogging Intensity Parameter Model</a></h4>
                    <p>The Two-Dimensional Otsu algorithm is commonly used for automatic thresholding in image segmentation. It helps separate foreground and background regions by finding an optimal threshold value. The Mixed Dark Channel algorithm enhances foggy images by segmenting them and adjusting dark channels based on fog density. An Adaptive Defogging Intensity Parameter Model dynamically adjusts the fogging investigative parameter to improve defogging, with the optimal value varying for different images. The model also estimates atmospheric light and computes the transmission map to achieve better restoration.</p>
                </div>

                <img src={R4}/>
                <div className="r4">
                    <h4><a href="https://ieeexplore.ieee.org/abstract/document/9511329" target="_blank">An Adaptive Defogging Intensity Parameter Model</a></h4>
                    <p>The Two-Dimensional Otsu algorithm is commonly used for automatic thresholding in image segmentation. It helps separate foreground and background regions by finding an optimal threshold value. The Mixed Dark Channel algorithm enhances foggy images by segmenting them and adjusting dark channels based on fog density. An Adaptive Defogging Intensity Parameter Model dynamically adjusts the fogging investigative parameter to improve defogging, with the optimal value varying for different images. The model also estimates atmospheric light and computes the transmission map to achieve better restoration.</p>
                </div>
            </div>
        </>
    )
}

export default Research;