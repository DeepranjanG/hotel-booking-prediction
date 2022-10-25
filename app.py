from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import HTMLResponse, RedirectResponse
from uvicorn import run as app_run

from hotel.constant.application import APP_HOST, APP_PORT
from hotel.pipeline.prediction_pipeline import (HotelData,HotelClassifier)
from hotel.pipeline.train_pipeline import TrainPipeline

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory='templates')

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DataForm:
    def __init__(self, request: Request):
        self.request: Request = request
        self.hotel: Optional[str] = None
        self.lead_time: Optional[str] = None
        self.arrival_date_year: Optional[str] = None
        self.arrival_date_month: Optional[str] = None
        self.arrival_date_day_of_month: Optional[str] = None
        self.stays_in_weekend_nights: Optional[str] = None
        self.stays_in_week_nights: Optional[str] = None
        self.meal: Optional[str] = None
        self.country: Optional[str] = None
        self.market_segment: Optional[str] = None
        self.distribution_channel: Optional[str] = None
        self.previous_cancellations: Optional[str] = None
        self.reserved_room_type: Optional[str] = None
        self.booking_changes: Optional[str] = None
        self.deposit_type: Optional[str] = None
        self.customer_type: Optional[str] = None
        self.adr: Optional[str] = None
        self.total_of_special_requests: Optional[str] = None
        self.total_guest: Optional[str] = None
        

    async def get_hotel_data(self):
        form = await self.request.form()
        self.hotel = form.get("hotel")
        self.lead_time = form.get("lead_time")
        self.arrival_date_year = form.get("arrival_date_year")
        self.arrival_date_month = form.get("arrival_date_month")
        self.arrival_date_day_of_month = form.get("arrival_date_day_of_month")
        self.stays_in_weekend_nights = form.get("stays_in_weekend_nights")
        self.stays_in_week_nights = form.get("stays_in_week_nights")
        self.meal = form.get("meal")
        self.country = form.get("country")
        self.market_segment = form.get("market_segment")
        self.distribution_channel = form.get("distribution_channel")
        self.previous_cancellations = form.get("previous_cancellations")
        self.reserved_room_type = form.get("reserved_room_type")
        self.booking_changes = form.get("booking_changes")
        self.deposit_type = form.get("deposit_type")
        self.customer_type = form.get("customer_type")
        self.adr = form.get("adr")
        self.total_of_special_requests = form.get("total_of_special_requests")
        self.total_guest = form.get("total_guest")

@app.get("/", tags=["authentication"])
async def index(request: Request):

    return templates.TemplateResponse(
            "hotel.html",{"request": request, "context": "Rendering"})


@app.get("/train")
async def trainRouteClient():
    try:
        train_pipeline = TrainPipeline()

        train_pipeline.run_pipeline()

        return Response("Training successful !!")

    except Exception as e:
        return Response(f"Error Occurred! {e}")


@app.post("/")
async def predictRouteClient(request: Request):
    try:
        form = DataForm(request)
        await form.get_hotel_data()
        
        hotel_data = HotelData(
                                hotel= form.hotel,
                                lead_time = form.lead_time,
                                arrival_date_year = form.arrival_date_year,
                                arrival_date_month = form.arrival_date_month,
                                arrival_date_day_of_month= form.arrival_date_day_of_month,
                                stays_in_weekend_nights= form.stays_in_weekend_nights,
                                stays_in_week_nights = form.stays_in_week_nights,
                                meal= form.meal,
                                country= form.country,
                                market_segment= form.market_segment,
                                distribution_channel=form.distribution_channel,
                                previous_cancellations=form.previous_cancellations,
                                reserved_room_type=form.reserved_room_type,
                                booking_changes=form.booking_changes,
                                deposit_type=form.deposit_type,
                                customer_type=form.customer_type,
                                adr=form.adr,
                                total_of_special_requests=form.total_of_special_requests,
                                total_guest=form.total_guest,
                                )
        
        hotel_df = hotel_data.get_hotel_input_data_frame()

        model_predictor = HotelClassifier()

        value = model_predictor.predict(dataframe=hotel_df)[0]

        status = None
        if value == 1:
            status = "Confirm"
        else:
            status = "Cancelled"

        return templates.TemplateResponse(
            "hotel.html",
            {"request": request, "context": status},
        )
        
    except Exception as e:
        return {"status": False, "error": f"{e}"}


if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)