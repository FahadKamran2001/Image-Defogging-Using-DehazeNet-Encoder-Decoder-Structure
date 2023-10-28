import React from 'react';
import { Carousel } from 'react-responsive-carousel';
import 'react-responsive-carousel/lib/styles/carousel.min.css';
import T1 from '../media/1.png';
import T2 from '../media/2.png';
import T3 from '../media/3.png';
import T4 from '../media/4.png';
import T5 from '../media/5.png';
import T6 from '../media/6.png';
import './Team.css';
import { AiFillLinkedin } from "react-icons/ai";


function Team() {
  const images = [
    
    {
      src: T6,
      alt: 'Image 6',
      name: 'Usama khan',
      status:"Mentor",
      profileLink:"https://www.linkedin.com/in/usama-khan-896b17160/"
    },
    
    {
      src: T2,
      alt: 'Image 2',
      name: 'Raheimeen Sherafgan',
      status:"Lead Developer",
      profileLink:"https://www.linkedin.com/in/raheimeen-sherafgan/"
    },
    {
      src: T1,
      alt: 'Image 1',
      name: 'Zain Ul Abideen',
      status:"Developer",
      profileLink:"https://www.linkedin.com/in/zainnisar/"
    },
    {
      src: T3,
      alt: 'Image 3',
      name: 'M.Abdur Rafey',
      status:"Developer",
      profileLink:"https://www.linkedin.com/in/m-a-rafey/"
    },
    {
      src: T4,
      alt: 'Image 4',
      name: 'Fahan Kamram',
      status:"Developer",
      profileLink:"https://www.linkedin.com/in/fahad-kamran-5b3373120/"
    },
    {
      src: T5,
      alt: 'Image 5',
      name: 'Dr. Hasan Mujhtaba',
      status:"Mentor",
      profileLink:"https://www.linkedin.com/in/hasan-mujtaba-994338236/"
    },
    
    
  ];

  return (
    <>
      <div className="teamContainer">
        <div className="teamInnerContainer">
        </div>
      </div>

      <div className='teamMainPage'>
        <h1>Our Team</h1>
      <div className="teamMembersContainer">
        <Carousel
          showStatus={false}
          showThumbs={false}
          infiniteLoop={true}
          autoPlay={true}
          showArrows={true}
          swipeable={true}
          centerMode={true}
          centerSlidePercentage={33.33}
          showIndicators ={false}
        >
          {images.map((image, index) => (
            <div key={index} className='carouselCard'>
              <img className="carouselImage" src={image.src} alt={image.alt} style={{width:"18rem"}}/>
              <h2 className='userName'>{image.name}</h2>
              <p className='status'>{image.status} <br></br><a href={image.profileLink} target='_blank' rel="noreferrer"><AiFillLinkedin size={35} className='profileLink'/></a></p>
              
            </div>
          ))}
        </Carousel>
      </div>

      </div>
    </>
  );
}

export default Team;
