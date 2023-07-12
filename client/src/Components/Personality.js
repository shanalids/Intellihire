import React from 'react'
import Navbar from './Navbar'
import Footer from './Footer'

const Personality = () => {
    return (
        <div>
            <Navbar />
            <div className="contact-form-container">
                <p>How often do you seek out new experiences or take risks?</p>
            </div>
            <div className="contact-form-container">
                <input type="text" placeholder="Answer" />
            </div>

            <div className="contact-form-container">
                <p>How do you ensure that your work is completed accurately and on time?</p>
            </div>
            <input type="text" placeholder="Answer" /><br /><br />

            <div className="contact-form-container">
                <p>How do you feel about social situations and making new connections?</p>
            </div>
            <input type="text" placeholder="Answer" /><br /><br />

            <div className="contact-form-container">
                <p>How do you handle feedback or criticism from others?</p>
            </div>
            <input type="text" placeholder="Answer" /><br /><br />

            <div className="contact-form-container">
                <p>Can you tell me about a time when you had to deal with a difficult or stressful situation? How did you handle it?</p>
            </div>
            <input type="text" placeholder="Answer" /><br /><br />

            <Footer />
        </div>
    )
}

export default Personality